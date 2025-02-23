from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("LAB4_01").getOrCreate()

rawdata = spark.read.csv('./assets/hour.csv', header=True)
rawdata.cache()

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])

new_schemaNames = new_rawdata.schema.names
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))

(trainingData, testData) = new_rawdata.randomSplit([0.7, 0.3], 42)

#---
# original default model
assembler = VectorAssembler(inputCols = new_schemaNames[0:new_ncolumns-3], outputCol = 'features')

glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01, family='poisson', link='log')

stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("original RMSE = %g " % rmse)

#---

# Onehot encoder
# 分类特征（One-Hot 编码）
categorical_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
# 数值特征（直接使用，不需编码）
numeric_features = ["temp", "atemp", "hum", "windspeed"]
# 目标变量（要预测的列）
label_col = "cnt"

# StringIndexer :features 2 INDEX
# indexers = []
# for col in categorical_features:
#     indexers.append(StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep"))
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
    for col in categorical_features
]

# OneHotEncoder :INDEX 2 One-Hot
encoders = [
    OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec")
    for col in categorical_features
]

# VectorAssembler :组合特征（包括 One-Hot 编码 & 数值特征）
assembler = VectorAssembler(
    inputCols=[col + "_vec" for col in categorical_features] + numeric_features,
    outputCol="features"
)

#--
# ONEHOT model
glm_poisson = GeneralizedLinearRegression(featuresCol="features", \
                                          labelCol=label_col, maxIter=50, \
                                          regParam=0.01, family="poisson", link="log")
stages_1H = indexers + encoders + [assembler, glm_poisson]
pipeline_1H = Pipeline(stages=stages_1H)

pipelineModel_1H = pipeline_1H.fit(trainingData)
predictions_1H = pipelineModel_1H.transform(testData)
evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
rmse_new = evaluator.evaluate(predictions_1H)

print("ONEHOT RMSE = %g " % rmse_new)

# model list
# solver :
# PySpark 3.x  LinearRegression removed owl-qn
# # 1. ℓ1 （Lasso）+ OWL-QN
# lr_l1_owlqn = LinearRegression(featuresCol="features", labelCol=label_col,
#                                regParam=0.01, elasticNetParam=1.0, solver="owl-qn")
# # 2. Elastic Net  + OWL-QN
# lr_elastic_owlqn = LinearRegression(featuresCol="features", labelCol=label_col,
#                                     regParam=0.01, elasticNetParam=0.5, solver="owl-qn")

# 1. ℓ1 （Lasso）+ OWL-QN
lr_l1_owlqn = LinearRegression(featuresCol="features", labelCol=label_col,
                               regParam=0.01, elasticNetParam=1.0, solver="l-bfgs")
# 2. Elastic Net  + OWL-QN
lr_elastic_owlqn = LinearRegression(featuresCol="features", labelCol=label_col,
                                    regParam=0.01, elasticNetParam=0.5, solver="l-bfgs")

# 3. ℓ2 （Ridge）+ L-BFGS
lr_l2_lbfgs = LinearRegression(featuresCol="features", labelCol=label_col,
                               regParam=0.01, elasticNetParam=0.0, solver="l-bfgs")
# 4. ℓ2 （Ridge）+ IRLS
glm_l2_irls = GeneralizedLinearRegression(featuresCol="features", labelCol=label_col,
                                          regParam=0.01, family="poisson", link="log", solver="irls")


models = {
    "L1 + OWL-QN": lr_l1_owlqn,
    "Elastic Net + OWL-QN": lr_elastic_owlqn,
    "L2 + L-BFGS": lr_l2_lbfgs,
    "L2 + IRLS": glm_l2_irls
}

evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")

for name, model in models.items():
    print(f"\nTraining model: {name}")

    # 组合 Pipeline
    stages_model = indexers + encoders + [assembler, model]
    pipeline = Pipeline(stages=stages_model)

    # 训练模型
    pipelineModel = pipeline.fit(trainingData)

    # 预测
    predictions = pipelineModel.transform(testData)

    # 计算 RMSE
    rmse = evaluator.evaluate(predictions)
    print(f"{name} RMSE = {rmse:.4f}")


spark.stop()