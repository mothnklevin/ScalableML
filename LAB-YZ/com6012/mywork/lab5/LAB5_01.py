from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.types import StringType


# 创建 SparkSession
spark = SparkSession.builder.appName("LAB5_01").getOrCreate()

# 读取数据
rawdata = spark.read.csv('./assets/spambase.data')
# 读取列名
spam_names = [line.rstrip('\n') for line in open('./assets/spambase.data.names')]


# 修正标签列名
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

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
trainData, testData = rawdata.randomSplit([0.7, 0.3], seed=1242)

# 组装特征向量
feature_cols = spam_names[:-1]  # 所有列，除了标签列
vecAssembler = VectorAssembler(inputCols=feature_cols, outputCol="features")



# 创建决策树分类器
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# 创建 Pipeline
pipeline = Pipeline(stages=[vecAssembler, dt])



# 创建参数网格
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [3, 5, 10])
             .addGrid(dt.maxBins, [16, 32, 64])
             .addGrid(dt.impurity, ["gini", "entropy"])
             .build())

# 评估器
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

# 交叉验证
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # 3折交叉验证

# 训练模型
Model = crossval.fit(trainData)

# 在测试数据上进行预测
predictions = Model.bestModel.transform(testData)


# 计算最佳模型的准确率
accuracy = evaluator.evaluate(predictions)
best_params = Model.bestModel.stages[-1]  # 获取最优决策树模型


print(f"bestModel maxDepth: {best_params.getMaxDepth()}")
print(f"bestModel maxBins: {best_params.getMaxBins()}")
print(f"bestModel impurity: {best_params.getImpurity()}")
print(f"bestModel accuracy: {accuracy:.4f}")

# 停止 SparkSession
spark.stop()
