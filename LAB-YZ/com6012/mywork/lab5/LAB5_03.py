from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.ml.evaluation import RegressionEvaluator

# 创建 SparkSession
spark = SparkSession.builder.appName("LAB5_03").getOrCreate()

# 读取 winequality 数据
rawdataw = spark.read.csv('./assets/winequality-white.csv', sep=';', header=True, inferSchema=True)

# 确保所有列为 double 类型
StringColumns = [x.name for x in rawdataw.schema.fields if x.dataType == StringType()]

for col_name in StringColumns:
    rawdataw = rawdataw.withColumn(col_name, col(col_name).cast("double"))
rawdataw = rawdataw.withColumnRenamed('quality', 'labels')


# 划分训练集和测试集
trainingData, testData = rawdataw.randomSplit([0.7, 0.3], seed=42)

# 获取所有特征列
feature_cols = [col for col in rawdataw.columns if col != 'labels']

# 组装特征
vecAssembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 创建随机森林回归器
rf = RandomForestRegressor(labelCol="quality", featuresCol="features", numTrees=100)

rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=10, numTrees=100,
                           featureSubsetStrategy="sqrt", seed=42, bootstrap=True)


# 训练模型
stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)
model = rf.fit(trainingData)

# 预测测试数据
predictions = model.transform(testData)

# 计算 RMSE 误差

evaluator = RegressionEvaluator(labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE = {rmse:.4f}")

# 获取特征重要性
# fi = model.featureImportances
feature_importance = model.stages[-1].featureImportances


# 转换为 NumPy 数组
imp_feat = np.zeros(len(feature_cols))
imp_feat[feature_importance.indices] = feature_importance.values

# 找出最重要的 3 个特征
sorted_indices = np.argsort(imp_feat)[::-1]  # 按重要性降序排序
top3_features = [(feature_cols[i], imp_feat[i]) for i in sorted_indices[:3]]

# 输出最重要的 3 个特征
print("\ntop3 importance features: ")
for rank, (feature, importance) in enumerate(top3_features, start=1):
    print(f"Top {rank} 特征: {feature}, 重要性: {importance:.4f}")

# 转换为 Pandas DataFrame 并排序
featureImp_df = pd.DataFrame(
    list(zip(feature_cols, imp_feat)), columns=["feature", "importance"]
)
featureImp_df = featureImp_df.sort_values(by="importance", ascending=False)

# 打印完整特征重要性表
print("\nall feature importance table: ")
print(featureImp_df)

# 停止 SparkSession
spark.stop()
