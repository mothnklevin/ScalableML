# 使用 PySpark 解析日志
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col
from pyspark.sql.functions import desc

# create Spark session
spark = SparkSession.builder.appName("LAB2_01_NASA_Log_Mining").getOrCreate()

print("\nLAB2_01_NASA_Log_Mining")

# def csv Schema
# StructType([StructField("name", DataType, nullable)])
log_schema = StructType([
    StructField("host", StringType(), True),
    StructField("logname", StringType(), True),
    StructField("user", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("request", StringType(), True),
    StructField("status", IntegerType(), True),
    StructField("bytes", StringType(), True)
])

# read NASA log
# df = spark.read\
#       .format("data_format")\     "csv"、"json"、"parquet"、"orc"、"jdbc"、"text"
#       .schema(schema_object) \    header,delimiter,inferSchema,nullValue,multiline,encoding,quote,escape,compression
#       .options(options)\
#       .load("path/to/file")\
df = spark.read \
    .option("delimiter", " ") \
    .schema(log_schema) \
    .csv("/users/acw24yz/com6012/mywork/lab1/NASA_access_log_Aug95.gz")
#  select useful 5 columns
df = df.select("host", "timestamp", "request", "status", "bytes")
# show first 5 column to check data
print("\nshow 'host, timestamp, request, status, bytes' schema: ")
df.show(5, truncate=False)


print("\n Part 1 Log mining: ")
# count unique hosts
unique_hosts = df.select("host")\
    .distinct()\
    .count()
print(f"\nTotal unique hosts in August 1995: {unique_hosts}")

# count most frequent visitor
most_frequent_visitor = df.groupBy("host")\
    .count()\
    .orderBy(desc("count"))\
    .limit(1)
print("\nmost frequent visitor: ")
most_frequent_visitor.show()

spark.stop()