from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NASA Log Analysis") \
    .config("spark.local.dir", "/mnt/parscratch/users/acw24yz") \
    .getOrCreate()

logFile = spark.read.text("NASA_access_log_Aug95.gz").cache()

totalRequests = logFile.count()

requestsFromTimken = logFile.filter(logFile.value.contains("gateway.timken.com")).count()

requestsOnAug15 = logFile.filter(logFile.value.contains("15/Aug/1995")).count()

errors404Total = logFile.filter(logFile.value.contains(" 404 ")).count()

errors404OnAug15 = logFile.filter((logFile.value.contains("15/Aug/1995")) & (logFile.value.contains(" 404 "))).count()

errors404FromTimkenOnAug15 = logFile.filter((logFile.value.contains("15/Aug/1995")) &
                                            (logFile.value.contains(" 404 ")) &
                                            (logFile.value.contains("gateway.timken.com"))).count()

print("\nNASA Log Analysis Results:")
print(f"1. Total requests: {totalRequests}")
print(f"2. Requests from gateway.timken.com: {requestsFromTimken}")
print(f"3. Requests on 15th August 1995: {requestsOnAug15}")
print(f"4. Total 404 errors: {errors404Total}")
print(f"5. 404 errors on 15th August: {errors404OnAug15}")
print(f"6. 404 errors from gateway.timken.com on 15th August: {errors404FromTimkenOnAug15}\n")


spark.stop()