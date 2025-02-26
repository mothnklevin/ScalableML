
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
import matplotlib
from pyspark.ml.classification import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("LAB5_teach").getOrCreate()


# ---
# part1. PySpark 中的决策树
print("\n part1")

rawdata = spark.read.csv('./assets/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./assets/spambase.data.names')]
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
# rawdata.printSchema()
print("打印原始特征类型1")
rawdata.describe().show()


# 将String变为Double类型
StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))

# 打印检查架构
# rawdata.printSchema()
print("打印原始特征类型2")
rawdata.describe().show()




# 创建训练集和测试集
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 1242)
# 打印检查行数
print(f"There are {trainingData.cache().count()} rows in the training set, and {testData.cache().count()} in the test set")



# 通过重新分区 DataFrame 来查看集群配置的效果
trainRepartitionData, testRepartitionData = (rawdata.repartition(24).randomSplit([0.7, 0.3], seed=1242))
print(trainRepartitionData.count())
# 因此,如果要修改训练和测试数据,
# 需要直接对randomsplit后的训练/测试集修改,
# 而不是重新执行randomsplit后祈祷拆分相同




# 使用 VectorAssembler连接特征
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')
vecTrainingData = vecAssembler.transform(trainingData)
vecTrainingData.select("features", "labels").show(5)

# 调整参数
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features", \
                            maxDepth=10, impurity='entropy')
model = dt.fit(vecTrainingData)

# 使用 featureImportance 获取特征的单个重要性
# 使用 values 和 indices 恢复 importance 的值和向量的索引，这些值不为零
fi = model.featureImportances
imp_feat = np.zeros(ncolumns-1)
imp_feat[fi.indices] = fi.values



# 绘制相对重要性图
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

x = np.arange(ncolumns-1)
plt.bar(x, imp_feat)
plt.savefig("./Output/feature_importances.png")

# 打印最重要特征
# spam_names[np.argmax(imp_feat)]
print("\n最重要的特征名称:")
print(spam_names[np.argmax(imp_feat)])



# 可视化 DecisionTree
print(model.toDebugString)



# 使用 Pandas 表格输出上述信息
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), model.featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)



# 测试集
# pipeline引入 VectorAssembler 和决策树
# Combine stages into pipeline
stages = [vecAssembler, dt]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

# 评估测试集的准确性
predictions = pipelineModel.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)




# ---

# part2. 集成方法
# part1是使用决策树进行分类的示例.
# 现在使用随机森林来执行回归
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# 预测葡萄酒的品质
rawdataw = spark.read.csv('./assets/winequality-white.csv', sep=';', header='true')
rawdataw.cache()

rawdataw.printSchema()

# 同样转换double并创建管道

StringColumns = [x.name for x in rawdataw.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdataw = rawdataw.withColumn(c, col(c).cast("double"))
rawdataw = rawdataw.withColumnRenamed('quality', 'labels')
# 打印检查
rawdataw.printSchema()

trainingDataw, testDataw = rawdataw.randomSplit([0.7, 0.3], 42)

vecAssemblerw = VectorAssembler(inputCols=StringColumns[:-1], outputCol="features")

rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=5, numTrees=3, \
                           featureSubsetStrategy = 'all', seed=123, bootstrap=False)
stages = [vecAssemblerw, rf]
pipeline = Pipeline(stages=stages)
pipelineModelw = pipeline.fit(trainingDataw)

predictions = pipelineModelw.transform(testDataw)

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

# 在葡萄酒质量数据集上使用 GBTRegressor
gbt = GBTRegressor(labelCol="labels", featuresCol="features", \
                   maxDepth=5, maxIter=5, lossType='squared', subsamplingRate= 0.5, seed=34)

# Create the pipeline
stages = [vecAssemblerw, gbt]
pipeline = Pipeline(stages=stages)
pipelineModelg = pipeline.fit(trainingDataw)

# Apply the pipeline to the test data
predictions = pipelineModelg.transform(testDataw)

evaluator = RegressionEvaluator \
      (labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)


spark.stop()