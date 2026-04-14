from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from delta import *
import os
from pyspark.sql.functions import col, upper, trim, when, regexp_replace, lit, explode, create_map
from config.spark_session import init_spark_session, create_spark_session


def load_mcc_mapping(spark: SparkSession, json_path: str):
    
    """
    Reads a flat JSON dictionary of MCC codes and converts it to a Spark DataFrame.
    """
    print(f"Loading MCC codes from {json_path}...")
    # with open(json_path, 'r') as f: // This is the traditional way to read JSON in Python, but it won't give us a Spark DataFrame directly.
    #     mcc_dict = json.load(f) // Problem using here with python 3.13 and Spark 3.4.1, so we will let Spark read the JSON directly.
    
    # Let Spark read it directly.
    raw_json_df = spark.read.option("multiline", "true").json(json_path)
    # Since it's a flat dictionary {"5812": "Restaurants", ...}, 
    # we can use PySpark SQL functions to pivot it into rows
    mcc_args = []
    
    for c in raw_json_df.columns:
        mcc_args.append(lit(c))
        mcc_args.append(col(c))
    
    df_exploded = raw_json_df.select(
        explode(create_map(*mcc_args)).alias("mcc", "category_name")
    )
    return df_exploded
    

def process_bronze_to_silver(spark: SparkSession, csv_path: str, json_path: str, output_path: str):
    """Reads raw Bronze data, enriches it with MCC categories, and writes to Silver Delta."""
    
    # 1. Load the Dimension Table (MCC Codes)
    mcc_df = load_mcc_mapping(spark, json_path)

    print(mcc_df.printSchema())
    # 2. Enforce the Schema on the Raw Data
    print(f"Reading raw transactions from {csv_path}...")
    
    # UPDATE THESE COLUMNS to match your ex.ipynb exactly!
    transaction_schema = StructType([
        StructField("id", StringType(), False),
        StructField("date", StringType(), True),
        StructField("client_id", StringType(), True), # The key to join users later
        StructField("card_id", StringType(), True),   # The key to join cards later
        StructField("amount", StringType(), True),    # READ AS STRING FIRST!
        StructField("use_chip", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("merchant_city", StringType(), True),
        StructField("merchant_state", StringType(), True),
        StructField("zip", StringType(), True),
        StructField("mcc", StringType(), True),
        StructField("errors", StringType(), True)
    ])

    raw_df = spark.read.csv(
        csv_path, 
        header=True, 
        schema=transaction_schema,
        mode="DROPMALFORMED"
    )

    # 3. Data Enrichment (The Join)
    print("Joining transactions with MCC Categories...")
    # We do a LEFT JOIN so we don't lose transactions if an MCC code is missing from the JSON
    enriched_df = raw_df.join(mcc_df, on="mcc", how="left")

    # 4. Data Cleaning
    print("Applying Silver layer transformations...")
    silver_df = enriched_df \
        .withColumn("amount", regexp_replace(col("amount"), "\\$", "").cast("double")) \
        .withColumn("category_name", when(col("category_name").isNull(), "UNKNOWN").otherwise(upper(trim(col("category_name"))))) \
        .filter((col("amount") < 1000000) & (col("amount") > -1000000)) \
        .dropna(subset=["id", "client_id", "amount"])
        

    # 5. Write to Silver Layer (Delta Lake)
    print(f"Writing ACID-compliant Delta Table to {output_path}...")
    silver_df.write \
        .format("delta") \
        .mode("overwrite") \
        .save(output_path)
        
    print("Bronze to Silver Delta pipeline completed successfully!")

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Define paths inside the Airflow container
    BRONZE_CSV_PATH = "/opt/airflow/data/bronze/transactions_data.csv"
    BRONZE_JSON_PATH = "/opt/airflow/data/bronze/mcc_codes.json" 
    SILVER_PATH = "/opt/airflow/data/silver/silver_transactions_delta"
    # BRONZE_CSV_PATH = "../../data/bronze/*.csv"
    # BRONZE_JSON_PATH = "../../data/bronze/mcc_codes.json" 
    # SILVER_PATH = "../../data/silver/silver_transactions_delta"
    # os.makedirs("../../data/silver", exist_ok=True)
    
    
    try:
        process_bronze_to_silver(spark, BRONZE_CSV_PATH, BRONZE_JSON_PATH, SILVER_PATH)
        # x = load_mcc_mapping(spark, BRONZE_JSON_PATH)
        
    finally:
        spark.stop()