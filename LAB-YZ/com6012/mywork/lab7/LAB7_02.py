from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType

import matplotlib.pyplot as plt

# 初始化 Spark 会话
spark = SparkSession.builder.appName("LAB7_02").getOrCreate()
print("LAB7_02 start")

schema_ratings = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", IntegerType(), True)
])
# 读取 CSV 文件并应用 Schema
ratings = spark.read.csv("Data/ml-latest-small/ratings.csv", header=True, schema=schema_ratings).cache()

schema_movies = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True)
])
# 读取电影数据 movies.csv
movies = spark.read.csv("Data/ml-latest-small/movies.csv", header=True, schema=schema_movies).cache()


# 准备训练/测试数据
myseed = 6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()

# 设定最佳 rank 值
best_rank = 15  # 假设最佳 rank 是 15

als = ALS(userCol="userId", itemCol="movieId", rank=best_rank, seed=myseed, coldStartStrategy="drop")
model = als.fit(training)

# 选择用户 ID
# selected_user = ratings.select("userId").distinct().limit(1).collect()[0]["userId"]
selected_user = 68  # 例如固定选择 userId = 10

# 前 5 个推荐电影
user_df = spark.createDataFrame([(selected_user,)], ["userId"])
recommendations = model.recommendForUserSubset(user_df, 5)

# 提取movieId
recommended_movie_ids = [row.movieId for row in recommendations.collect()[0]["recommendations"]]

# 电影标题和类型
recommended_movies = movies.filter(movies.movieId.isin(recommended_movie_ids))
recommended_movies_list = recommended_movies.select("title", "genres").collect()

# 打印用户 ID 和推荐的电影信息
print(f"TOP 5 MOVIES for user {selected_user}  :")
for movie in recommended_movies_list:
    print(f"movie: {movie['title']}, genres: {movie['genres']}")

print("LAB7_02  end")

spark.stop()