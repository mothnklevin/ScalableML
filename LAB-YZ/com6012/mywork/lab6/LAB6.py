# part1. PySpark 中的浅层神经网络

# 在 Spark 中启用 Arrow
# Enable Arrow-based columnar data transfers. This line of code will be explained later
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# 使用实验3中的spambase数据集
# 加载数据集并正确设置相应的 DataFrame
# We load the dataset and the names of the features
import numpy as np

rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)

spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# We rename the columns in the dataframe with names of the features in spamd.data.names
schemaNames = rawdata.schema.names
spam_names[ncolumns - 1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# We cast the type string to double
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))

# 创建训练集和测试集
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 42)

# 为向量汇编器和神经网络创建实例
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')

# MultilayerPerceptronClassifier实现神经网络模型
# Spark ML只允许在中间层中使用 sigmoid 激活函数，
#               在输出层中使用 softmax 函数
# 通过参数指定神经网络架构,例如:layers=[10, 5, 4, 3]
    # 第一层有 10 个节点（特征），
    # 然后是两个隐藏层（5 个和 4 个节点）
    # 以及最后一层 3 个输出（类）
from pyspark.ml.classification import MultilayerPerceptronClassifier
# The first element HAS to be equal to the number of input features
layers = [len(trainingData.columns)-1, 20, 5, 2]
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", maxIter=100, layers=layers, seed=1500)

# 创建管道，拟合到数据，并计算测试集的性能
# Create the pipeline
from pyspark.ml import Pipeline
stages = [vecAssembler, mpc]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)

# We now make predictions
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)


#---
# part2. PySpark 上的 Keras

# Spark ML 仅允许sigmoid 激活函数,
# 要使用更先进的模型,一种方法是使用pandas UDF
# 或使用不同的panda函数API. 如 mapInPandas（）
# pyspark使用 Apache Arrow在JVM与Python间传输数据

# 本部分使用 spambase 数据集的子集训练 Keras 模型
# 然后使用 mapInPandas（） 要求执行程序计算测试集的预测

# 使用 keras 在 Spambase 数据集上训练神经网络模型
#启用 Arrow 以转换dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# 将 spark 数据帧转换为 pandas 数据帧
# Convert the Spark DataFrame to a Pandas DataFrame using Arrow
trainingDataPandas = trainingData.select("*").toPandas()

# 为 keras 准备数据
nfeatures = ncolumns-1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures]
ytrain = trainingDataPandas.iloc[:, -1]

# 配置神经网络模型
# 在 HPC 上运行时,安装 TensorFlow会收到GPU配置警告
# 本实验内可忽略警告
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(np.shape(Xtrain)[1],)))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 使用 20% 的训练数据对数据进行拟合以进行验证
history = model.fit(Xtrain, ytrain, epochs=100, batch_size=100, validation_split=0.2, verbose=False)

# # 可视化
# # 绘制训练进度图
# import matplotlib.pyplot as plt
#
# history_dict = history.history
# acc_values= history_dict['accuracy']
# val_acc_values= history_dict['val_accuracy']
# epochs = range(1, len(acc_values)+1)
#
# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("./Output/keras_nn_train_validation_history.png")



# 现在，将使用 mapInPandas 来有效地分配计算
# 为了能够将训练好的模型广播给 executor，模型需要是 picklable的
# 也就是说，要求模型可以被 pickle 模块序列化
# 不幸的是，Keras 不支持 pickle 序列化其对象（模型）
# Zach Moshe 提出了一个用于序列化 Keras 模型的补丁
# 我们使用 StackOverFlow 条目中建议的类ModelWrapperPickable
import tempfile
import tensorflow
import pandas as pd

class ModelWrapperPickable:

    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        model_str = ''
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tensorflow.keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            self.model = tensorflow.keras.models.load_model(fd.name)

# 使用类ModelWrapperPickable封装 Keras 模型
model_wrapper = ModelWrapperPickable(model)

# 此处 mapInPandas（）返回一个具有相同列特征的 DataFrame 和一个用于预测的附加列


# 从仅包含特征的数据帧中提取testData ，不包含 labels
Xtest = testData.select(spam_names[0:ncolumns-1])

# 输出数据帧的新架构
from pyspark.sql.types import StructField, StructType, DoubleType
pred_field = [StructField("prediction", DoubleType(), True)]
new_schema = StructType(Xtest.schema.fields + pred_field)

# 创建方法来计算预测
def predict(iterator):
    for features in iterator:
        yield pd.concat(\
            [features, \
             pd.Series(model_wrapper.model.predict(features).flatten(), \
                       name="prediction")], axis=1)

# 使用 mapInPandas 将predict应用于dataframe的Xtest批次
prediction_keras_df = Xtest.mapInPandas(predict, new_schema)

# 生成的 DataFrame 是 Spark DataFrame
# 选择预测列并将其转换为 pandas
ypred_keras = prediction_keras_df.select('prediction').toPandas().values

# 使用阈值 0.5 将预测分配给类 0 和类 1
ypred_keras[ypred_keras <0.5] = 0
ypred_keras[ypred_keras >0.5] = 1

# 从 DataFrame 中提取目标测试数据testData
testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values

# 使用 scikit-learn 的 accuracy_score 方法来计算准确率
from sklearn.metrics import accuracy_score
print("Accuracy = %g " % accuracy_score(ypred_keras, ytest))



#---
# PART3. 练习

# 练习 1
# 在第 1 节中,
# 为应用于 spambase 的神经网络管道添加交叉验证步骤
# 使paramGrid 包含参数layers的不同值，
# 并在测试数据上查找最佳参数和关联的准确率

# 练习 2
# 重复本实验的[第2节](2- keras -on- pyspark)
# 但现在使用Scikit-learn模型进行实验
# 选择一个可用的分类器,在相同的训练数据上训练分类器，
# 并使用pandas和arrow将模型发送给执行器，执行器将提供预测。
# 你需要使用“ModelWrapperPickable”类吗？
