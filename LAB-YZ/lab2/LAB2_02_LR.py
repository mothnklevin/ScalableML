import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, monotonically_increasing_id


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 创建 SparkSession
spark = SparkSession.builder.appName("LAB2_02_AD_LR").getOrCreate()
# print("LAB2_02_AD_LR")
logging.info("\nLAB2_02_AD_LR Spark session started.")


# 读取数据
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/users/acw24yz/com6012/ScalableML/Data/Advertising.csv")     # use absolute path

df2 = df.drop('_c0')
# print("show 5 df :")
# df2.show(5)
logging.info("Data loaded and `_c0` column dropped.")

# def transData(data):
#     return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])
# transformed= transData(df2)
# print("show 5 df2 : ")
# transformed.show(5)
assembler = VectorAssembler(inputCols=["TV", "radio", "newspaper"], outputCol="features")
df2 = assembler.transform(df2).select("features", "sales")
df2 = df2.withColumnRenamed("Sales", "label")
logging.info("df2 transformed into ML format")


# data set split
(trainingData, testData) = df2.randomSplit([0.6, 0.4], 6012)

# lr = LinearRegression()
# lrModel = lr.fit(trainingData)
#
# Add regularisation
models = {
    "No Regularization": LinearRegression(regParam=0.0),
    "L2 Reg(Ridge)": LinearRegression(regParam=0.05,elasticNetParam=0.0),
    "L1 Reg(Lasso)": LinearRegression(regParam=0.05, elasticNetParam=1.0),
    "Mixed Reg(Elastic Net)": LinearRegression(regParam=0.05, elasticNetParam=0.5),
    # additional:
    "L2 Reg-enhanced": LinearRegression(regParam=0.4, elasticNetParam=0.0),
    "L1 Reg-enhanced": LinearRegression(regParam=0.4, elasticNetParam=1.0),
    "Mixed Reg-enhanced": LinearRegression(regParam=0.4, elasticNetParam=0.5)
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
    # predictions[name].select("features", "label", "prediction").show(5)


evaluator = RegressionEvaluator(labelCol="label", \
                                predictionCol="prediction", \
                                metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
rmse_results = {}
for name, pred in predictions.items():
    rmse = evaluator.evaluate(pred)
    rmse_results[name] = rmse
    print(f"\n {name} - RMSE: {rmse:.4f}")
# dict 2 list of tuples
rmse_results = [(name, rmse) for name, rmse in rmse_results.items()]

rmse_df = spark.createDataFrame(rmse_results, ["Model", "RMSE"])
rmse_df = rmse_df.orderBy(col("RMSE").asc())
# withColumn create new column
rmse_df = rmse_df.withColumn("Rank", monotonically_increasing_id() + 1)
rmse_df.show()



spark.stop()
logging.info("Spark session stopped.")

