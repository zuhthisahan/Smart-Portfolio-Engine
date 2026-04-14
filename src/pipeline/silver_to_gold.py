from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, countDistinct, when, abs as _abs, 
    date_format, lit, round as _round, greatest, least
)
from delta import *
from config.spark_session import create_spark_session

def process_silver_to_gold(spark: SparkSession, users_path: str, txn_path: str, output_path: str):
    
    print("Loading Silver Delta Tables...")
    user_df = spark.read.format('delta').load(users_path)
    txn_df = spark.read.format('delta').load(txn_path)
    
    # Feature Extractions
    # 1. Behavioural Aggreegations (From Transactions)
    
    print("Calculating Behavioral Spending Features...")
    expenses_df = txn_df.filter(col('amount') <0 ) \
        .withColumn('expense_amount', _abs(col('amount'))) \
        .withColumn('txn_month', date_format(col('date'), 'yyyy-MM'))
    
    # Define what counts as "Discretionary" (Wants vs Needs)
    # Define what counts as "Discretionary" (Wants vs Needs)
    # THESE MUST MATCH YOUR mcc_codes.json EXACTLY (in uppercase)
    discretionary_categories = [
        "EATING PLACES AND RESTAURANTS", 
        "FAST FOOD RESTAURANTS", 
        "DRINKING PLACES (ALCOHOLIC BEVERAGES)",
        "AMUSEMENT PARKS, CARNIVALS, CIRCUSES", 
        "MOTION PICTURE THEATERS", 
        "BOOK STORES", 
        "DEPARTMENT STORES",
        "DISCOUNT STORES",
        "SHOE STORES",
        "COSMETIC STORES",
        "MISCELLANEOUS HOME FURNISHING STORES",
        "GIFT, CARD, NOVELTY STORES",
        "POTTERY AND CERAMICS",
        "LEATHER GOODS",
        "TRAVEL AGENCIES", 
        "LODGING - HOTELS, MOTELS, RESORTS", 
        "PASSENGER RAILWAYS",
        "RAILROAD PASSENGER TRANSPORT",
        "GARDENING SUPPLIES"
    ]
    
    tagged_expenses_df = expenses_df.withColumn(
        'is_discretionary',
        when(col('category_name').isin(discretionary_categories), col('expense_amount')).otherwise(lit(0))
    )
    
    # Aggregate spending by user-level features
    txn_features = tagged_expenses_df.groupBy('client_id') \
        .agg(
            _sum('expense_amount').alias('total_expenses'),
            _sum('is_discretionary').alias('discretionary_spending'),
            countDistinct("txn_month").alias("active_months")
        )
    # Monthly averages
    user_spend_profiles = txn_features \
        .withColumn("avg_monthly_spend", col("total_expenses") / col("active_months")) \
        .withColumn("avg_monthly_discretionary", col("discretionary_spending") / col("active_months")) \
        .drop("total_expenses", "discretionary_spending", "active_months")
        
    # 2. Joined with User Demographics
    print("Engineering Final ML Features...")
    
    gold_df = user_df.join(user_spend_profiles, user_df.id == user_spend_profiles.client_id, how='inner') \
        .withColumn('monthly_income', col('yearly_income') / 12) \
        .withColumn('investment_horizon', col("retirement_age") - col("current_age")) \
        .withColumn('credit_utilization_ratio', _round(col('total_debt') / col('yearly_income'), 4)) \
        .withColumn('monthly_disposable_income', _round(col('monthly_income') - col('avg_monthly_spend'))) \
        .withColumn('savings_rate', _round(col('monthly_disposable_income') / col('monthly_income'), 4)) \
        .withColumn("discretionary_spend_ratio", _round(col("avg_monthly_discretionary") / col("avg_monthly_spend"), 4))
    
    # 3. Risk Tolerance score (1-10)
    # A simple rule-based expert system: Base score 5, adjusted by age and debt.
    
    base_score = lit(5)
    
    risk_score_calc = base_score \
        + when(col("investment_horizon") > 20, 2).otherwise(0) \
        - when(col("investment_horizon") < 10, 2).otherwise(0) \
        + when(col("credit_score") > 750, 1).otherwise(0) \
        - when(col("credit_utilization_ratio") > 0.4, 2).otherwise(0)
        
    final_gold_df = gold_df \
        .withColumn('risk_tolerance_score', greatest(lit(1), least(lit(10), risk_score_calc)))
    
    # 4. Write to Gold Delta Table
    print(f"Writing Gold Delta Table to {output_path}...")
    final_gold_df.write.format('delta').mode('overwrite').save(output_path)
    
    print("Silver to Gold pipeline completed successfully! Ready for the Optimizer.")
        
if __name__ == "__main__":
    spark = create_spark_session()
    
    # Paths inside Airflow
    SILVER_USERS_PATH = "/opt/airflow/data/silver/silver_users_delta"
    SILVER_TXN_PATH = "/opt/airflow/data/silver/silver_transactions_delta"
    GOLD_PATH = "/opt/airflow/data/gold/user_portfolio_features"
    
    # Local paths for testing
    # SILVER_USERS_PATH = "../../data/silver/silver_user_delta"
    # SILVER_TXN_PATH = "../../data/silver/silver_transactions_delta"
    # GOLD_PATH = "../../data/gold/user_portfolio_features"
    try:
        process_silver_to_gold(spark, SILVER_USERS_PATH, SILVER_TXN_PATH, GOLD_PATH)
    finally:
        spark.stop()