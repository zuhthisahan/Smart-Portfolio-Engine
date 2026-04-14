from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum as _sum, abs as _abs, date_format, lag, avg, round as _round
from config.spark_session import create_spark_session

def build_time_series_features(spark, txn_path, output_path):
    
    print("Loading Silver Transactions...")
    txn_df = spark.read.format("delta").load(txn_path)
    
    # 1. Filter for expenses and extract the month
    monthly_spend_df = txn_df.filter(col("amount") < 0) \
        .groupBy("client_id", date_format(col("date"), "yyyy-MM").alias("txn_month")) \
        .agg(_sum(_abs(col("amount"))).alias("current_month_spend"))
    
    # 3. Time-Series Feature Engineering (Lags and Rolling Averages)
    # Define a Window specification: Partition by User, Order by Month
    
    user_window = Window.partitionBy("client_id").orderBy("txn_month")
    rolling_3m_window = Window.partitionBy("client_id").orderBy("txn_month").rowsBetween(-3, -1)
    
    ts_features_df = monthly_spend_df \
        .withColumn("spend_1_month_ago", _round(lag("current_month_spend", 1).over(user_window), 2)) \
        .withColumn("spend_2_months_ago", _round(lag("current_month_spend", 2).over(user_window), 2)) \
        .withColumn("spend_3_months_ago", _round(lag("current_month_spend", 3).over(user_window), 2)) \
        .withColumn("avg_spend_last_3_months", _round(avg("current_month_spend").over(rolling_3m_window), 2))
    
    # drop rows with nulls 
    ts_features_df = ts_features_df.dropna(subset = ["current_month_spend"])
    
    # 5. Write to a new Gold Delta Table
    print(f"Writing Time-Series Gold Table to {output_path}...")
    ts_features_df.write.format("delta").mode("overwrite").save(output_path)
    print("Time-Series Gold table built successfully!")
    

if __name__ == "__main__":
    spark = create_spark_session()
    SILVER_TXN_PATH = "/opt/airflow/data/silver/silver_transactions_delta"
    GOLD_TS_PATH = "/opt/airflow/data/gold/time_series_features"
    
    try:
        build_time_series_features(spark, SILVER_TXN_PATH, GOLD_TS_PATH)
    finally:
        spark.stop()