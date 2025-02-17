from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LAB3_01").getOrCreate()

trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np

schemaNames = trainingData.schema.names  # 读取数据框的列名
ncolumns = len(schemaNames)
spam_names = schemaNames[:]  # 重新定义 spam_names

vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')

# no reg
lr = LogisticRegression(featuresCol='features', \
                        labelCol='labels', maxIter=50, \
                        regParam=0, family="binomial")

stages = [vecAssembler, lr]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)

predictions = pipelineModel.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("no_reg Accuracy = %g " % accuracy)

w_no_reg = pipelineModel.stages[-1].coefficients.values

lrMode = pipelineModel.stages[-1]
AC_T_NO = lrMode.summary.accuracy
print("NO_Reg Training Accuracy = %g " % AC_T_NO)


# L1
lrL1 = LogisticRegression(featuresCol='features', \
                          labelCol='labels', maxIter=50, \
                          regParam=0.01, elasticNetParam=1, \
                          family="binomial")
stageslrL1 = [vecAssembler, lrL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(trainingData)

predictions = pipelineModellrL1.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("L1 Accuracy = %g " % accuracy)
w_L1 = pipelineModellrL1.stages[-1].coefficients.values

lrModeL1 = pipelineModellrL1.stages[-1]
AC_T_L1 = lrModeL1.summary.accuracy
print("L1 Training Accuracy = %g " % AC_T_L1)


# L2
lrL2 = LogisticRegression(featuresCol='features', \
                          labelCol='labels', maxIter=50, \
                          regParam=0.01, elasticNetParam=0, \
                          family="binomial")
stageslrL2 = [vecAssembler, lrL2]
pipelinelrL2 = Pipeline(stages=stageslrL2)
pipelineModellrL2 = pipelinelrL2.fit(trainingData)

predictions = pipelineModellrL2.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("L2 Accuracy = %g " % accuracy)
w_L2 = pipelineModellrL2.stages[-1].coefficients.values

lrModeL2 = pipelineModellrL2.stages[-1]
AC_T_L2 = lrModeL2.summary.accuracy
print("L2 Training Accuracy = %g " % AC_T_L2)


# Elastic Net reg
lrL3 = LogisticRegression(featuresCol='features', \
                          labelCol='labels', maxIter=50, \
                          regParam=0.01, elasticNetParam=0.5, \
                          family="binomial")
stageslrL3 = [vecAssembler, lrL3]
pipelinelrL3 = Pipeline(stages=stageslrL3)
pipelineModellrL3 = pipelinelrL3.fit(trainingData)

predictions = pipelineModellrL3.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("EN Accuracy = %g " % accuracy)
w_L3 = pipelineModellrL3.stages[-1].coefficients.values

lrModeL3 = pipelineModellrL3.stages[-1]
AC_T_L3 = lrModeL3.summary.accuracy
print("EN Training Accuracy = %g " % AC_T_L3)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(10, 8))
ax1.plot(w_no_reg)
ax1.set_title('No regularisation')
ax2.plot(w_L1)
ax2.set_title('L1 regularisation')
ax3.plot(w_L2)
ax3.set_title('L2 regularisation')
ax4.plot(w_L3)
ax4.set_title('EN regularisation')

plt.tight_layout()
plt.savefig("./Output/all4_reg.png")
print("save Vec pic")

# spam_names[np.argmax(np.abs(w_no_reg))]
# spam_names[np.argmax(np.abs(w_L1))]
# spam_names[np.argmax(np.abs(w_L2))]
# spam_names[np.argmax(np.abs(w_L3))]

print("no_reg highest feature: ", \
      spam_names[np.argmax(np.abs(w_no_reg))] )
print("L1_reg highest feature: ", \
      spam_names[np.argmax(np.abs(w_L1))] )
print("L2_reg highest feature: ", \
      spam_names[np.argmax(np.abs(w_L2))] )
print("EN_reg highest feature: ", \
      spam_names[np.argmax(np.abs(w_L3))] )

spark.stop()