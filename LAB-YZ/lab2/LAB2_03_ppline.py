from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

spark = SparkSession.builder.appName("LAB2_03_ppline").getOrCreate()

print("LAB2_03_ppline")

training = spark.createDataFrame([
    (0, "a b c d e spark 6012", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h 6012", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

training.printSchema()
training.show()

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(training)

# test = spark.createDataFrame([
#     (4, "spark i j 6012"),
#     (5, "l m n"),
#     (6, "spark 6012 spark"),
#     (7, "apache hadoop")
# ], ["id", "text"])
new_test = spark.createDataFrame([
    (4, "pyspark hadoop"),
    (5, "spark a b c"),
    (6, "mapreduce spark"),
], ["id", "text"])

new_test.show()

prediction = model.transform(new_test)
prediction.show()

selected = prediction.select("id", "text", "probability", "prediction")
selected.show()

for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

spark.stop()
