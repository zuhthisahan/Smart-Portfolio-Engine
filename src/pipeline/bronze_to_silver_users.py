from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql import SparkSession
from delta import *
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from config.spark_session import create_spark_session

def process_users(spark: SparkSession, input_path: str, output_path: str):
    print(f"Reading raw user data from {input_path}...")
    
    # Define User Schema
    user_schema = StructType([
        StructField("id", StringType(), False),
        StructField("current_age", IntegerType(), True),
        StructField("retirement_age", IntegerType(), True),
        StructField("birth_year", IntegerType(), True),
        StructField("birth_month", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("address", StringType(), True),
        StructField("latitude", FloatType(), True),
        StructField("longitude", FloatType(), True),
        StructField("per_capita_income", StringType(), True),
        StructField("yearly_income", StringType(), True), # Read as string due to $ signs
        StructField("total_debt", StringType(), True),    # Read as string due to $ signs
        StructField("credit_score", IntegerType(), True),
        StructField('num_credit_cards', IntegerType(), True)
    ])
    
    raw_user_df = spark.read.csv(input_path, header=True, schema=user_schema, mode="DROPMALFORMED")
    # display(raw_user_df.show(5, truncate=False))
    # Clean the Financial Columns (Strip '$' and convert to Numbers)
    print('Applying Silver layer transformations for Users...')
    silver_user_df = raw_user_df \
        .withColumn('yearly_income', regexp_replace(col('yearly_income'), "\\$", "").cast('double')) \
        .withColumn('total_debt', regexp_replace(col('total_debt'), "\\$", "").cast('double')) \
        .withColumn('per_capita_income', regexp_replace(col('per_capita_income'), "\\$", "").cast('double')) \
        .dropna(subset=['id', 'yearly_income'])
        
    # write to delta table
    print(f"Writing Users Delta Table to {output_path}...")
    # display(silver_user_df.show(5), truncate=False)
    silver_user_df.write.format('delta').mode('overwrite').save(output_path)
    
    print('User data processed successfully!')
    
    

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Paths map to Airflow container volumes
    BRONZE_USERS_PATH = "/opt/airflow/data/bronze/users_data.csv"
    SILVER_USERS_PATH = "/opt/airflow/data/silver/silver_users_delta"
    
    # BRONZE_USERS_PATH = "../../data/bronze/users_data.csv"
    # SILVER_USERS_PATH = "../../data/silver/silver_user_delta"
    
    try:
        process_users(spark, BRONZE_USERS_PATH, SILVER_USERS_PATH)
    finally:
        spark.stop()
    