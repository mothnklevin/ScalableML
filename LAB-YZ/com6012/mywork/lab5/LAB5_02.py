from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

# 创建 SparkSession
spark = SparkSession.builder.appName("LAB5_02").getOrCreate()

# 读取数据
rawdata = spark.read.csv('./assets/spambase.data')

# 读取列名
spam_names = [line.rstrip('\n') for line in open('./assets/spambase.data.names')]

# 处理列名，去除 `:` 及后续部分
# number_names = np.shape(spam_names)[0]
# for i in range(number_names):
#     local = spam_names[i]
#     colon_pos = local.find(':')
#     spam_names[i] = local[:colon_pos]

spam_names = [name.split(':')[0] for name in spam_names]

# 修正标签列名
ncolumns = len(rawdata.columns)
spam_names[ncolumns-1] = 'label'

# 重命名 DataFrame 列
schemaNames = rawdata.schema.names
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# 转换所有字符串列为 double
string_columns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for col_name in string_columns:
    rawdata = rawdata.withColumn(col_name, col(col_name).cast("double"))

# 划分训练集和测试集
trainData, testData = rawdata.randomSplit([0.7, 0.3], seed=42)

# 组装特征向量
feature_cols = rawdata.columns[:-1]  # 选择所有列，除了 `label`
vecAssembler = VectorAssembler(inputCols=feature_cols, outputCol="features")


# 创建随机森林分类器
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# 创建 Pipeline
pipeline = Pipeline(stages=[vecAssembler, rf])


# 创建超参数网格
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [5, 10, 15])  # 决策树最大深度
             .addGrid(rf.maxBins, [16, 32, 64])  # 分桶数
             .addGrid(rf.numTrees, [10, 50, 100])  # 随机森林树的数量
             .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt", "log2"])  # 选择特征的策略
             .addGrid(rf.subsamplingRate, [0.7, 1.0])  # 采样比例
             .build())

# 评估器（使用准确率）
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

# 交叉验证
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # 3 折交叉验证

# 训练交叉验证模型
Model = crossval.fit(trainData)

# 在测试数据上进行预测
predictions = Model.bestModel.transform(testData)

# 计算最佳模型的测试集准确率
accuracy = evaluator.evaluate(predictions)
best_rf = Model.bestModel.stages[-1]  # 获取最佳的随机森林模型

# 输出最佳参数和测试集上的准确率
print(f"bestModel maxDepth: {best_rf.getMaxDepth()}")
print(f"bestModel maxBins: {best_rf.getMaxBins()}")
print(f"bestModel numTrees: {best_rf.getNumTrees()}")
print(f"bestModel featureSubsetStrategy: {best_rf.getFeatureSubsetStrategy()}")
print(f"bestModel subsamplingRate: {best_rf.getSubsamplingRate()}")
print(f"bestModel accuracy: {accuracy:.4f}")

# 停止 SparkSession
spark.stop()
