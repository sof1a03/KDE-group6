import pandas as pd
import requests
import pyspark
import numpy as np
import pickle
import base64
import pandas as pd
import os
from multiprocessing.pool import ThreadPool
from pyspark.sql.types import StringType, IntegerType, BinaryType, DoubleType, ArrayType, StructType, StructField
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, expr, lit, sum as spark_sum
from datetime import datetime
from graphframes import GraphFrame
from scipy.sparse import csr_matrix, vstack, hstack
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


books = pd.read_csv("Books.csv")

books['ISBN'] = books['ISBN'].str.upper()

books.info()

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


from pyspark.sql.functions import (
    col,
    udf,
    row_number,
    countDistinct,
    collect_list,
    struct,
    count,
    sum,
    avg,
    expr,
    lit,
    percentile_approx,
    max as spark_max,
    explode,
    least,
    greatest
)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ISBNValidationandAPICheck") \
    .master("local[*]") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
    .config("spark.executor.memory", "20G") \
    .config("spark.driver.memory", "50G") \
    .config("spark.executor.memoryOverhead", "1G") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.driver.maxResultSize", "2G") \
    .getOrCreate()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, expr, lit, sum as spark_sum

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Corrected ISBN Validation Debugged") \
    .getOrCreate()

# Load the dataset
books = spark.read.csv("Books.csv", header=True, inferSchema=True)

# Normalize ISBN column: Remove hyphens and whitespace
books = books.withColumn("ISBN", expr("regexp_replace(trim(ISBN), '-', '')"))

# Validate ISBN Length (10 or 13)
books = books.withColumn(
    "is_valid_length",
    when((length(col("ISBN")) == 10) | (length(col("ISBN")) == 13), True).otherwise(False)
)

# Add position column (POS) for each character in ISBN
books_with_pos = books.withColumn("POS", expr("sequence(1, length(ISBN))"))
books_exploded = books_with_pos.withColumn("POS", expr("explode(POS)"))

# Extract character values using substring
books_exploded = books_exploded.withColumn(
    "digit_value",
    when(expr("substring(ISBN, POS, 1)") == "X", lit(10))
    .otherwise(expr("CAST(substring(ISBN, POS, 1) AS INT)"))
)

# Add weights for ISBN-10 (Corrected Formula)
books_exploded = books_exploded.withColumn(
    "weight_10",
    when(length("ISBN") == 10, 11 - col("POS")).otherwise(0)
)

# Add weights for ISBN-13
books_exploded = books_exploded.withColumn(
    "weight_13",
    when(length("ISBN") == 13, when((col("POS") % 2) == 1, 1).otherwise(3)).otherwise(0)
)

# Compute weighted contributions for ISBN-10 and ISBN-13
books_exploded = books_exploded.withColumn(
    "weighted_value_10",
    col("digit_value") * col("weight_10")
).withColumn(
    "weighted_value_13",
    col("digit_value") * col("weight_13")
)

# Sum contributions for ISBN-10 and ISBN-13
weighted_sum = books_exploded.groupBy("ISBN").agg(
    spark_sum("weighted_value_10").alias("weighted_sum_10"),
    spark_sum("weighted_value_13").alias("weighted_sum_13"),
    spark_sum(
        when(col("POS") == 10, col("digit_value")).otherwise(0)
    ).alias("last_digit")
)

# Add checksum validations for ISBN-10 and ISBN-13
weighted_sum = weighted_sum.withColumn(
    "checksum_valid_10",
    (length(col("ISBN")) == 10) & (((col("weighted_sum_10")) % 11) == 0)
).withColumn(
    "checksum_valid_13",
    (length(col("ISBN")) == 13) & ((col("weighted_sum_13") % 10) == 0)
)

# Combine checksum validations
weighted_sum = weighted_sum.withColumn(
    "Validity",
    when(col("checksum_valid_10") | col("checksum_valid_13"), "Valid").otherwise("Invalid")
)

# Debugging: Show intermediate results for ISBN 0671002481
debug_weights = books_exploded.filter(col("ISBN") == "0671002481")
debug_weights.select("ISBN", "POS", "digit_value", "weight_10", "weighted_value_10").show(truncate=False)

debug_sum = weighted_sum.filter(col("ISBN") == "0671002481")
debug_sum.select("ISBN", "weighted_sum_10", "last_digit", "checksum_valid_10", "Validity").show(truncate=False)

# Show all ISBNs with their validity status
weighted_sum.select("ISBN", "Validity").show(truncate=False)

# Stop Spark Session
spark.stop()

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Corrected ISBN Validation Debugged") \
    .getOrCreate()

# Load the dataset
books = spark.read.csv("Books.csv", header=True, inferSchema=True)

# Normalize ISBN column: Remove hyphens and whitespace
books = books.withColumn("ISBN", expr("regexp_replace(trim(ISBN), '-', '')"))

# Validate ISBN Length (10 or 13)
books = books.withColumn(
    "is_valid_length",
    when((length(col("ISBN")) == 10) | (length(col("ISBN")) == 13), True).otherwise(False)
)

# Add position column (POS) for each character in ISBN
books_with_pos = books.withColumn("POS", expr("sequence(1, length(ISBN))"))
books_exploded = books_with_pos.withColumn("POS", expr("explode(POS)"))

# Extract character values using substring
books_exploded = books_exploded.withColumn(
    "digit_value",
    when(expr("upper(substring(ISBN, POS, 1))") == "X", lit(10))
    .otherwise(expr("CAST(substring(ISBN, POS, 1) AS INT)"))
)

# Add weights for ISBN-10 (Corrected Formula)
books_exploded = books_exploded.withColumn(
    "weight_10",
    when(length("ISBN") == 10, 11 - col("POS")).otherwise(0)
)

# Add weights for ISBN-13
books_exploded = books_exploded.withColumn(
    "weight_13",
    when(length("ISBN") == 13, when((col("POS") % 2) == 1, 1).otherwise(3)).otherwise(0)
)

# Compute weighted contributions for ISBN-10 and ISBN-13
books_exploded = books_exploded.withColumn(
    "weighted_value_10",
    col("digit_value") * col("weight_10")
).withColumn(
    "weighted_value_13",
    col("digit_value") * col("weight_13")
)

# Sum contributions for ISBN-10 and ISBN-13
weighted_sum = books_exploded.groupBy("ISBN").agg(
    spark_sum("weighted_value_10").alias("weighted_sum_10"),
    spark_sum("weighted_value_13").alias("weighted_sum_13"),
    spark_sum(
        when(col("POS") == 10, col("digit_value")).otherwise(0)
    ).alias("last_digit")
)

# Add checksum validations for ISBN-10 and ISBN-13
weighted_sum = weighted_sum.withColumn(
    "checksum_valid_10",
    (length(col("ISBN")) == 10) & (((col("weighted_sum_10")) % 11) == 0)
).withColumn(
    "checksum_valid_13",
    (length(col("ISBN")) == 13) & ((col("weighted_sum_13") % 10) == 0)
)

# Combine checksum validations
weighted_sum = weighted_sum.withColumn(
    "Validity",
    when(col("checksum_valid_10") | col("checksum_valid_13"), "Valid").otherwise("Invalid")
)

# Join back with the original dataset to retrieve titles
books_with_validity = books.join(weighted_sum, on="ISBN", how="inner")

# Filter for invalid ISBNs only
invalid_books = books_with_validity.filter(col("Validity") == "Invalid")

# Select and show ISBNs and titles of invalid books
invalid_books.select("ISBN", "Book_Title", "Book_Author").show(truncate=False)
# Save invalid books to a CSV file
invalid_books.write.csv("invalid_books.csv", header=True, mode="overwrite")
# Stop Spark Session
spark.stop()

