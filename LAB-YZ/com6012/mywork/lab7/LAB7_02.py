from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
import matplotlib.pyplot as plt

# 初始化 Spark 会话
spark = SparkSession.builder.appName("LAB7_02").getOrCreate()

# 读入数据并拆分单词（逗号分隔）
lines = spark.read.text("Data/ml-latest-small/ratings.csv").rdd
parts = lines.map(lambda row: row.value.split(","))

# 手动指定数据格式并转换为 DataFrame（去除 timestamp 列）
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD).cache()

# 读取电影数据 movies.csv
movies_lines = spark.read.text("Data/ml-latest-small/movies.csv").rdd
movies_parts = movies_lines.map(lambda row: row.value.split(","))
moviesRDD = movies_parts.map(lambda p: Row(movieId=int(p[0]), title=p[1], genres=p[2]))
movies = spark.createDataFrame(moviesRDD).cache()

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
print(f"推荐给用户 {selected_user} 的前 5 部电影:")
for movie in recommended_movies_list:
    print(f"电影: {movie['title']}, 类型: {movie['genres']}")

spark.stop()