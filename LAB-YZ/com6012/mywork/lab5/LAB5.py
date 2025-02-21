# part1. PySpark 中的决策树

# 加载数据集及其特征与标签
# 同时缓存dataframe
import numpy as np
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# 使用withColumnRenamed方法重命名列
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

#打印原始特征的类型
rawdata.printSchema()

# 使用cast()，
# 将String变为Double类型
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))

# 打印检查架构
rawdata.printSchema()

# 创建训练集和测试集
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 1242)
# 打印检查行数
print(f"There are {trainingData.cache().count()} rows in the training set, and {testData.cache().count()} in the test set")

# 在集群中,即使使用相同种子,由于集群配置可能更改,
# randomSplit仍可能导致不同的训练集和测试集
# 数据集小时,基本可以在一个分区中.
# 可以通过重新分区 DataFrame 来查看集群配置的效果
trainRepartitionData, testRepartitionData = (rawdata.repartition(24).randomSplit([0.7, 0.3], seed=1242))
print(trainRepartitionData.count())
# 因此,如果要修改训练和测试数据,
# 需要直接对randomsplit后的训练/测试集修改,
# 而不是重新执行randomsplit后祈祷拆分相同

# 使用 VectorAssembler连接特征
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')
vecTrainingData = vecAssembler.transform(trainingData)
vecTrainingData.select("features", "labels").show(5)

# 调整参数
# maxDepth：树的最大深度。默认值为 5。
# maxBins：从连续要素创建多少个 bin。默认值为 32。
# impurity：计算信息增益的度量。选项包括 “gini” 或 “entropy”。默认值为 “gini”
# minInfoGain：用于拆分的最小信息增益。默认值为 0。
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features", \
                            maxDepth=10, impurity='entropy')
model = dt.fit(vecTrainingData)

# 使用 featureImportance 获取特征的单个重要性
# 使用 values 和 indices 恢复 importance 的值和向量的索引，这些值不为零
fi = model.featureImportances
imp_feat = np.zeros(ncolumns-1)
imp_feat[fi.indices] = fi.values

# 绘制相对重要性图
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
x = np.arange(ncolumns-1)
plt.bar(x, imp_feat)
plt.savefig("./Output/feature_importances.png")

# 打印最重要特征
spam_names[np.argmax(imp_feat)]

# 可视化 DecisionTree
print(model.toDebugString)

# 使用 Pandas 表格输出上述信息
import pandas as pd
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), model.featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)

# 也可以使用 spark-tree-plotting
# 将Spark 树转换为 JSON 格式;
# 随后使用 D3 对其进行可视化，
# 也可以从 JSON 转换为 DOT 并使用 graphviz


# 测试集
# pipeline引入 VectorAssembler 和决策树
from pyspark.ml import Pipeline
# Combine stages into pipeline
stages = [vecAssembler, dt]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

# 评估测试集的准确性
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

# part2. 集成方法
# Spark 中的随机森林: bagging 或 bootstrap 聚合
# 分类:预测是通过在单个树之间进行多数投票
# 回归:预测是每棵树的单个预测的平均值

# part1是使用决策树进行分类的示例.
# 现在使用随机森林来执行回归

# 预测葡萄酒的品质
rawdataw = spark.read.csv('./Data/winequality-white.csv', sep=';', header='true')
rawdataw.cache()

rawdataw.printSchema()

# 同样转换double并创建管道
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdataw.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdataw = rawdataw.withColumn(c, col(c).cast("double"))
rawdataw = rawdataw.withColumnRenamed('quality', 'labels')
# 打印检查
rawdataw.printSchema()

trainingDataw, testDataw = rawdataw.randomSplit([0.7, 0.3], 42)

vecAssemblerw = VectorAssembler(inputCols=StringColumns[:-1], outputCol="features")

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=5, numTrees=3, \
                           featureSubsetStrategy = 'all', seed=123, bootstrap=False)
stages = [vecAssemblerw, rf]
pipeline = Pipeline(stages=stages)
pipelineModelw = pipeline.fit(trainingDataw)

predictions = pipelineModelw.transform(testDataw)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)

pipelineModelw.stages[-1].featureImportances

#表格化输出重要性
featureImp = pd.DataFrame(
  list(zip(vecAssemblerw.getInputCols(), pipelineModelw.stages[-1].featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)



# 梯度提升:集成中的每棵树按顺序训练

# PySpark 使用GBTRegressor实现用于回归的梯度提升树，
# 使用类GBTClassifier来实现用于二元分类的梯度提升树。
# 尚未为多类分类实现 GBT

# 在葡萄酒质量数据集上使用 GBTRegressor
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(labelCol="labels", featuresCol="features", \
                   maxDepth=5, maxIter=5, lossType='squared', subsamplingRate= 0.5, seed=34)

# Create the pipeline
stages = [vecAssemblerw, gbt]
pipeline = Pipeline(stages=stages)
pipelineModelg = pipeline.fit(trainingDataw)

# Apply the pipeline to the test data
predictions = pipelineModelg.transform(testDataw)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator \
      (labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)



#---
# part3. 练习

# 练习 1

# 为[应用于spambase数据集的决策树分类器]的管道
# 包含一个交叉验证步骤
# 使paramGrid包含maxDepth,maxBins,impurity 的不同值,
# 并在测试数据上找到最佳参数和相关精度

# 练习 2
# 将 RandomForestClassifier 应用于 spambase 数据集
# 包括一个交叉验证步骤，其中包含一个 paramGrid
# 其中包含 maxDepth、maxBins、numTrees、
# featureSubsetStrategy 和 subsamplingRate 选项,
# 在测试数据上找到最佳参数和相关的准确性


# 练习 3
# 使用 featureImportances 方法
# 研究随机森林中每个特征的相对重要性
# 使用用于 wine 数据集的随机森林回归器中的 featureImportances，
# 并指示三个最相关的特征。
# 如何计算特征重要性？

