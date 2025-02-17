from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LAB3_02").getOrCreate()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')

schemaNames = trainingData.schema.names
ncolumns = len(schemaNames)
spam_names = schemaNames[:]

vecAssembler = VectorAssembler(inputCols=spam_names[0:ncolumns-1], outputCol='features')

lr = LogisticRegression(featuresCol='features', labelCol='labels', \
                        maxIter=50, family="binomial")

# ParamGridBuilder
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.05, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [1.0, 0.0, 0.5]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

# CrossValidator
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Pipeline
pipeline = Pipeline(stages=[vecAssembler, crossval])
Model = pipeline.fit(trainingData)
predictions = Model.transform(testData)


best_accuracy = evaluator.evaluate(predictions)
print(f"Best Model Accuracy: {best_accuracy:.4f}")

# 选择最佳正则化参数
best_model = Model.stages[-1].bestModel
best_regParam = best_model._java_obj.getRegParam()
best_elasticNetParam = best_model._java_obj.getElasticNetParam()

print(f"Best regParam: {best_regParam}")
print(f"Best elasticNetParam: {best_elasticNetParam}")

# additional：
# 不同超参数组合的评估得分
crossval_model = Model.stages[-1]  # 获取 CrossValidatorModel

print("\n Hyperparameter combinations and accuracy：")
for i, params in enumerate(crossval_model.getEstimatorParamMaps()):
    regParam = params[best_model.regParam]
    elasticNetParam = params[best_model.elasticNetParam]
    accuracy = crossval_model.avgMetrics[i]  # 获取accuracy
    print(f" regParam: {regParam}, elasticNetParam: {elasticNetParam}, Accuracy: {accuracy:.4f}")

spark.stop()