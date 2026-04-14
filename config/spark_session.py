from delta import *
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os

def init_spark_session():
    python_path = r"D:\Data Engineering\new_env\Scripts\python.exe"
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

    builder = SparkSession.builder \
        .appName("FootballLakehouse_Silver_Players") \
        .master("local[*]") \
        .config("spark.pyspark.python", python_path) \
        .config("spark.pyspark.driver.python", python_path) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark Session Started.")
    return spark

def create_spark_session():
    """Initialize a local Spark Session optimized for Docker and Delta Lake."""
    
    builder = SparkSession.builder \
        .appName('SmartPortfolio-BronzeToSilver-Delta') \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") 
    
        
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark Session Started.")
    
    return spark