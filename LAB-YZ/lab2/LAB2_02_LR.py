

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# 创建 SparkSession
spark = SparkSession.builder.appName("LAB2_02_AD_LR").getOrCreate()

print("LAB2_02_AD_LR")

# 读取数据
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/users/acw24yz/com6012/ScalableML/Data/Advertising.csv")     # use absolute path

df2 = df.drop('_c0')
print("show 5 df :")
df.show(5)

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])
transformed= transData(df2)
print("show 5 df2 : ")
transformed.show(5)

# data set split
(trainingData, testData) = transformed.randomSplit([0.6, 0.4], 6012)

# lr = LinearRegression()
# lrModel = lr.fit(trainingData)

# Add regularisation
models = {
    "No Regularization": LinearRegression(regParam=0.0),
    "L2 Regularization (Ridge)": LinearRegression(regParam=0.1,elasticNetParam=0.0),
    "L1 Regularization (Lasso)": LinearRegression(regParam=0.1, elasticNetParam=1.0),
    "Mixed regularization (Elastic Net)": LinearRegression(regParam=0.1, elasticNetParam=0.5),
    # additional:
    "L2 Regularization-enhanced": LinearRegression(regParam=0.4, elasticNetParam=0.0),
    "L1 Regularization-enhanced": LinearRegression(regParam=0.4, elasticNetParam=1.0),
    "Mixed regularization-enhanced": LinearRegression(regParam=0.4, elasticNetParam=0.5)
}

trained_models = {}
for name, lr in models.items():
    print(f"\nTraining {name} Model")
    trained_models[name] = lr.fit(trainingData)
    print(f"Training {name} Model successfully")


# predictions = lrModel.transform(testData)
# predictions.show(5)
predictions = {}
for name, model in trained_models.items():
    print(f"\nEvaluating {name} Model")
    predictions[name] = model.transform(testData)
    print(f"Evaluating {name} Model successfully")
    # optional ,test
    predictions[name].select("features", "label", "prediction").show(5)


evaluator = RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
rmse_results = {}
for name, pred in predictions.items():
    rmse = evaluator.evaluate(pred)
    rmse_results[name] = rmse
    print(f"{name} - RMSE: {rmse:.4f}")


spark.stop()
