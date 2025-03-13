# PART1: k-means 聚类
# class pyspark.ml.clustering.KMeans(featuresCol='features', predictionCol='prediction', k=2, initMode='k-means||', initSteps=2, tol=0.0001, maxIter=20, seed=None, distanceMeasure='euclidean', weightCol=None)

# 请求内核
srun --pty --cpus-per-task=2 bash -i

# 安装：matplotlib 并进入shell
pip install matplotlib
pyspark --master local[2]

# 启用pyplot：
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

# 导入模块
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt

# 简单合成数据的聚类
# 对具有四个完美分离的数据点的简单示例进行k-means聚类
data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
        (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
df = spark.createDataFrame(data, ["features"])
kmeans = KMeans(k=2, seed=1)  # Two clusters with seed = 1
model = kmeans.fit(df)

# 检查聚类中心（质心
# 使用经过训练的模型来 “预测” 数据点的聚类索引
centers = model.clusterCenters()
len(centers)
# 2
for center in centers:
    print(center)
# [0.5 0.5]
# [8.5 8.5]
model.predict(df.head().features)
# 0

# 使用模型对数据点进行聚类
transformed = model.transform(df)
transformed.show()

# 检查经过训练的模型的训练摘要
model.hasSummary
# True
summary = model.summary
summary
# <pyspark.ml.clustering.KMeansSummary object at 0x2b1662948d30>
summary.k
# 2
summary.clusterSizes
# [2, 2]]
summary.trainingCost  #sum of squared distances of points to their nearest center
# 2.0

# 保存和加载算法/模型

# 保存并加载k·均值算法
import tempfile

temp_path = tempfile.mkdtemp()
kmeans_path = temp_path + "/kmeans"
kmeans.save(kmeans_path)
kmeans2 = KMeans.load(kmeans_path)
kmeans2.getK()

# 保存并加载已学习的k-means 模型（不包括摘要）
model_path = temp_path + "/kmeans_model"
model.save(model_path)
model2 = KMeansModel.load(model_path)
model2.hasSummary
# False
model2.clusterCenters()
# [array([0.5, 0.5]), array([8.5, 8.5])]

# 鸢尾花聚类

# 加载和检查数据
df = spark.read.load("Data/iris.csv", format="csv", inferSchema="true", header="true").cache()
df.show(5,True)

df.printSchema()

# 检查（统计数据）数据
df.describe().show()

# 使用transData将（features）数据转换为密集向量
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])

dfFeatureVec= transData(df).cache()
dfFeatureVec.show(5, False)

# 确定k通过轮廓分析
import numpy as np

numK=10
silhouettes = np.zeros(numK)
costs= np.zeros(numK)
for k in range(2,numK):  # k = 2:9
    kmeans = KMeans().setK(k).setSeed(11)
    model = kmeans.fit(dfFeatureVec)
    predictions = model.transform(dfFeatureVec)
    costs[k]=model.summary.trainingCost
    evaluator = ClusteringEvaluator()  # to compute the silhouette score
    silhouettes[k] = evaluator.evaluate(predictions)

# 查看聚类结果
predictions.show(15)

# 绘制成本（k点到其最近质心的平方距离之和，越小越好）
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,numK),costs[2:numK],marker="o")
ax.set_xlabel('$k$')
ax.set_ylabel('Cost')
plt.grid()
plt.savefig("Output/Lab8_cost.png")

# 绘制 silhouette 度量（越大越好）
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,numK),silhouettes[2:numK],marker="o")
ax.set_xlabel('$k$')
ax.set_ylabel('Silhouette')
plt.grid()
plt.savefig("Output/Lab8_silhouette.png")




# PART2. 练习

# 1. 选择k=3并使用 scikit-learn 中提供的归一化互信息 （NMI）
# 根据地面实况（class 标签）评估聚类结果。
# 您需要通过 conda install -y scikit-learn
# 在 myspark 环境中安装scikit-learn。
# 这使我们能够在知道聚类的真实数量时研究聚类质量。




# 2. 使用多个（例如，10 或 20 个）随机种子生成不同的聚类结果，
# 并绘制相应的 NMI 值（相对于地面实况k=3,如问题1所示）\
# 来观察初始化的效果。

