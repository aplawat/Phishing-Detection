# spark_processor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, lit
from pyspark.sql.types import StringType, IntegerType
import re

def extract_domain(url):
    """Extract domain from URL"""
    try:
        return re.findall(r'https?://([^/]+)', url)[0]
    except:
        return ""

def count_special_chars(url):
    """Count special characters in URL"""
    special_chars = ['@', '?', '-', '_', '%', '.', '=', '&']
    return sum(url.count(char) for char in special_chars)

def preprocess_data(data):
    """Preprocess raw data and create features"""
    # Initialize Spark session
    spark = SparkSession.builder.appName("PhishingPreprocessor").getOrCreate()
    
    # Convert raw data to DataFrame
    df = spark.createDataFrame(data)
    
    # Extract features
    extract_domain_udf = udf(extract_domain, StringType())
    count_special_chars_udf = udf(count_special_chars, IntegerType())
    
    # Apply transformations
    processed_df = df.withColumn("domain", extract_domain_udf(col("url"))) \
                    .withColumn("special_char_count", count_special_chars_udf(col("url"))) \
                    .withColumn("url_length", col("url").cast("string").length()) \
                    .withColumn("is_https", when(col("url").startswith("https://"), 1).otherwise(0))
    
    # For demonstration purposes, create a binary label
    # In a real system, you would use actual labeled data
    processed_df = processed_df.withColumn(
        "label", 
        when(
            (col("domain").contains(".ru")) | 
            (col("domain").contains(".xyz")) | 
            (col("url").contains("login")) | 
            (col("special_char_count") > 10), 
            1
        ).otherwise(0)
    )
    
    return processed_df