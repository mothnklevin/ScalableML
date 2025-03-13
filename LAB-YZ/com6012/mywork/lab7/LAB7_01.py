from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import matplotlib.pyplot as plt

# 初始化 Spark 会话
spark = SparkSession.builder.appName("LAB7_01").getOrCreate()
print("LAB7_01 start")

schema_ratings = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", IntegerType(), True)
])

# 读取 CSV 文件并应用 Schema
ratings = spark.read.csv("Data/ml-latest-small/ratings.csv", header=True, schema=schema_ratings).cache()

# 准备训练/测试数据
myseed = 6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()

# 设定不同的 rank 值
rank_values = [5, 10, 15, 20, 25]
rmse_values = []

for rank in rank_values:
    print(f"Training ALS model with rank = {rank}")
    als = ALS(userCol="userId", itemCol="movieId", rank=rank, seed=myseed, coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    print(f"Root-mean-square error (RMSE) for rank {rank}: {rmse}")
    rmse_values.append(rmse)

# 画出 RMSE vs Rank 的折线图
plt.figure(figsize=(8, 5))
plt.plot(rank_values, rmse_values, marker='o', linestyle='-', color='b')
plt.xlabel("Rank")
plt.ylabel("RMSE")
plt.title("Rank - RMSE")
plt.grid()
plt.savefig("RMSEvsRank.PNG")
plt.show()

print("LAB7_01 end")

spark.stop()