# PART1. 通过协同过滤推荐电影
# 请求2内核
srun --pty --cpus-per-task=2 bash -i
# 启动 PySpark shell
pyspark --master local[2]

# 协同过滤：基于矩阵本身来填充用户-项目关联矩阵的缺失条目
# 交替最小二乘法 （ALS） 算法：
# class pyspark.ml.recommendation.ALS(*, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False, alpha=1.0, userCol='user', itemCol='item', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=10, intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='nan', blockSize=4096)

# 电影推荐
# 数据集格式：“ user id | item id | rating | timestamp ”
#               196      242        3      881250949
#   显式反馈与隐式反馈
#   ratings 作为条目为显式反馈，
#   如果使用点击次数或观影累计时间等，则为隐式反馈
#   冷启动问题：测试集中的某些用户和/或项目不存在的情况

# 电影镜头100k 中的ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# 读入数据并拆分单词（制表符分隔）
lines = spark.read.text("Data/MovieLens100k.data").rdd
parts = lines.map(lambda row: row.value.split("\t"))

# 将文本 （String） 转换为数字 int 或 float ，然后将 RDD 转换为 DataFrame
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD).cache()

# # 检查数据
# ratings.show(5)
#
# # 检查数据类型：
# ratings.printSchema()

# 准备训练/测试数据
myseed=6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()

# 使用 ALS 对训练数据构建推荐模型
    # 将冷启动策略设置为drop 以确保我们不会获得 NaN 评估指标
    # 会有 BLAS 和 LAPACK 警告
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model = als.fit(training)

# 计算测试数据的 RMSE 来评估模型
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))
# # Root-mean-square error = 0.9209573069829078

# 为每个用户生成前 10 部电影推荐
userRecs = model.recommendForAllUsers(10)
userRecs.show(5,  False)

# 为每部电影生成前 10 名用户推荐
movieRecs = model.recommendForAllItems(10)
movieRecs.show(5, False)

# 为一组指定的用户生成前 10 部电影推荐
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
users.show()
userSubsetRecs.show(3,False)

# 为一组指定的电影生成前 10 名用户推荐
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movies.show()
movieSubSetRecs.show(3,False)


# 看看学习的因素
dfItemFactors=model.itemFactors
dfItemFactors.show()


#--
# part2 ：练习

# 1. 使用wget将实验室 2 练习中的 MovieLens ml-latest-small 数据集
# 下载到 HPC 上的目录ScalableML/Data。
# 使用unzip命令将文件解压缩到您选择的目录（搜索“unzip linux”以查看使用示例）。
# 阅读此数据集的自述文件以了解数据。
#
# 2. 使用ALS在此数据集上学习五个推荐模型，
# 使用与上述相同的拆分比率 （0.8, 0.2） 和 seed（6012），
# 但rank参数使用五个不同值：5、10、15、20、25。
# 将 5 个生成的 RMSE 值（在测试集上）与 5 个rating值作图。
#
#
# 3. 找到要推荐给您选择的任何一个用户的前五部电影，
# 并显示这五部电影的标题和类型（通过编程）。




