from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import time
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

spark = SparkSession.builder.appName("LAB6_01").getOrCreate()

start_time = time.time()
print("\nEX1 START")

# 加载数据
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)

# 读取特征名称
spam_names = [line.rstrip('\n') for line in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# 重命名列
schemaNames = rawdata.schema.names
spam_names[ncolumns - 1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# 将所有列转换为 double 类型


stringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in stringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))

# 拆分数据集
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 42)


# 特征工程
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')










# 定义神经网络结构
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", maxIter=100, seed=1500)



# 创建 Pipeline
stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)

# 交叉验证和参数搜索
paramGrid = ParamGridBuilder() \
    .addGrid(mpc.layers, [
        [len(trainingData.columns) - 1, 10, 2],  # 1
        [len(trainingData.columns) - 1, 15, 10, 5, 2],  # 2
        [len(trainingData.columns) - 1, 20, 5, 2],  # 2
        [len(trainingData.columns) - 1, 30, 15, 5, 2],  # 3
    ]) \
    .build()

# 评估器
evaluator = MulticlassClassificationEvaluator(
    labelCol="labels", predictionCol="prediction", metricName="accuracy"
)

# 交叉验证设置
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3  # 3 折交叉验证
)

# 训练模型
Model = crossval.fit(trainingData)

# 在测试集上评估
predictions = Model.transform(testData)
accuracy = evaluator.evaluate(predictions)

# 获取最佳模型的 pipeline
best_model = Model.bestModel
# 获取 pipeline 中的所有 stages
best_stages = best_model.stages

# 获取所有参数组合的准确率
param_maps = Model.getEstimatorParamMaps()
avg_metrics = Model.avgMetrics  # 存储所有参数组合的交叉验证得分

# 遍历所有超参数组合
print("\nall layer config and accuracy :")
for i, param in enumerate(param_maps):
    layers = param[mpc.layers]
    print(f"  - layers: {layers} , training Accuracy: {avg_metrics[i]:.4f}")

# 遍历 pipeline，找到 MultilayerPerceptronClassifier
best_mlp_model = None
for stage in best_stages:
    if isinstance(stage, MultilayerPerceptronClassificationModel):  # 检查是否为 MLP 模型
        best_mlp_model = stage
        break  # 找到 MLP 直接跳出循环

# 确保找到了 MLP 模型
if best_mlp_model:
    best_layers = best_mlp_model.getLayers()  # 获取最佳参数
    print(f"best parameters (layers): {best_layers}")
else:
    print("Error: Could not find MLP model in pipeline!")

print(f"best parameters(stages[-1] directly): {Model.bestModel.stages[-1].layers}")
print(f"best accuracy (test): {accuracy:.4f}")













end_time = time.time()
running_time = end_time - start_time
print(f"\nex 1 running time: {running_time:.2f} sec")



#---
#---

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql.types import StructField, StructType, DoubleType
from pyspark.sql.functions import pandas_udf
import pickle

import time
start_time = time.time()
print("\nEX2 START")

# 启用 Arrow 加速 Pandas 和 Spark 之间的数据传输
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# 将 Spark DataFrame 转换为 Pandas DataFrame
trainingDataPandas = trainingData.select("*").toPandas()

# 提取特征和标签
nfeatures = ncolumns-1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures]
ytrain = trainingDataPandas.iloc[:, -1]

# 训练 Scikit-learn 模型（随机森林分类器）
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(Xtrain, ytrain)

# 由于 Scikit-learn 支持 Pickle，因此 **不需要 ModelWrapperPickable**
model_bytes = pickle.dumps(rf_model)

# 从 testData 提取特征
Xtest = testData.select(spam_names[:-1])

# 定义新架构（增加预测列）
pred_field = [StructField("prediction", DoubleType(), True)]
new_schema = StructType(Xtest.schema.fields + pred_field)

# 定义 `predict` 函数
def predict(iterator):
    # 反序列化模型
    model = pickle.loads(model_bytes)
    for features in iterator:
        preds = model.predict(features)
        yield pd.concat([features, pd.Series(preds, name="prediction")], axis=1)

# 使用 `mapInPandas()` 进行分布式预测
prediction_sklearn_df = Xtest.mapInPandas(predict, new_schema)

# 提取预测值并转换为 NumPy 数组
ypred_sklearn = prediction_sklearn_df.select('prediction').toPandas().values

# 提取测试集真实标签
testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values

# 计算准确率
accuracy = accuracy_score(ytest, ypred_sklearn)
print(f"Scikit-learn RandomForest Accuracy: {accuracy:.4f}")


end_time = time.time()
running_time = end_time - start_time
print(f"\nex 2 running time: {running_time:.2f} sec")


spark.stop()