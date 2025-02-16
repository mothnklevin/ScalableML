# lab3：可扩展的Logistic回归

# 安装 matplotlib pandas
pip install matplotlib pandas
# 只需在环境中安装 matplotlib 和 pandas 一次

# part1. Spark 配置
# LAB1中，在`/mnt/parscratch/users/YOUR_USERNAME`设置了spark.local.dir
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/mnt/parscratch/users/acw24yz") \
    .getOrCreate()
# spark.local.dir的属性用作“暂存scratch”空间

# 通过以下方式检查自定义的（未显示默认值）配置
sc._conf.getAll()

# 驱动程序内存问题
# .sh内存：
    # 从 Stanage 请求的内存要求在 .sh 文件中的以下两行中配置
    #     !/bin/bash
    #     SBATCH --cpus-per-task=n n指定最大核数
    #     SBATCH --mem-per-cpu=nG n指定最大内存量
    # 上述配置，获得2*8= 16GB实际内存
    # 实验室队列使用`node209` 和 `node210`，
    # 每个大节点共有 1024 GB 内存和 64 个节点，
    #   即每个节点64GB。
    # 不要请求超出配置的内存
# 驱动内存
# 使用`spark-submit`，默认驱动程序内存仅为 1G
# 注意`spark.driver.memory`或`spark.executor.memory`等需要请求内存的设置
#   在文件中指定配置，程序配置内存量不应超过sh环境允许的内存量
#   例如：
#     #SBATCH --cpus-per-task=10
#     #SBATCH --mem-per-cpu=8G
#     spark-submit --driver-memory 10g --executor-memory 8g ./Code/LogMiningBig.py
#   8*10 = 80G小于1024G，sh环境正确
#   spark-submit 请求10G > 8G，因此程序没有可用的节点

# worker 线程的数量
#   SparkSession.builder.master("local[2]") \ ...
# 但也可以使用 spark-submit 指定 worker 线程的数量
# spark-submit --driver-memory 5g --executor-memory 5g --master local[10] ./Code/LogMiningBig.py
# 注意: SparkSession.builder指定的现成必须与spark-submit相同,
# 否则: Spark.Conf直接设置 > spark-submit > spark-defaults.conf
# 最后将使用SparkSession.builder中的配置
# 最后: Session线程数 应当 ≤ sh请求节点数
#如果作业的实际内存使用量>请求的内核/节点数，则作业将被终止
#示例
# spark-submit --driver-memory 20g --executor-memory 20g --master local[10] --local.dir /mnt/parscratch/users/USERNAME --conf spark.driver.maxResultSize=4g test.py


#---

# part2. PySpark 中的 Logistic 回归
#---
#   1. 文件设置
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

spam_names[spam_names.index('char_freq_;')] = 'char_freq_semicolon'
spam_names[spam_names.index('char_freq_(')] = 'char_freq_leftp'

# 使用更熟悉的特征名称重命名列
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

# 从 pyspark.sql.types 导入 Double 类型
#   对 dataframe 使用 withColumn 方法，
#   并将列 cast（） 为 DoubleType
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(spam_names[i], rawdata[spam_names[i]].cast(DoubleType()))

# 使用在上一个Notebook中使用的相同种子将数据拆分为 training 和 test
(trainingDatag, testDatag) = rawdata.randomSplit([0.7, 0.3], 42)

# 使用 Apache Parquet 格式,将这两个集保存到磁盘
# 以便以后使用,例如:比较不同转换与数据或 ML 模型的性能
trainingDatag.write.mode("overwrite").parquet('./Data/spamdata_training.parquet')
testDatag.write.mode("overwrite").parquet('./Data/spamdata_test.parquet')

# 从磁盘中读取两个文件
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')

# 创建 VectorAssembler 以连接向量中的所有特征
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features')

#---
# 2. Logistic 回归
    # maxIter：最大迭代次数
    # regParam：正则化参数 （≥0)
    # elasticNetParam：ElasticNet 的 mixing 参数。
    #   它采用 [0,1] 范围内的值。
    #   为α=0，则惩罚为ℓ2. 为α=1，则惩罚为ℓ1.
    # family：二项式（二元分类）或多项式（多类分类）。它也可以是 'auto'。
    # standardization：是否在拟合模型之前标准化训练特征。
    #   它可以是 true 或 false（默认为 True）。

# 在相同的训练数据上训练不同的分类器

#---
# 从 Logistic 回归开始，没有正则化，所以λ=0
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0, family="binomial")

# 创建一个pipeline，并将其拟合到训练数据中
from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [vecAssembler, lr]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)

# 计算一下精度
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

# 保存无正则化的向量vector w
w_no_reg = pipelineModel.stages[-1].coefficients.values

#---
# 现在仅使用ℓ1正则化 （λ=0.01和α=1)
lrL1 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=1, family="binomial")

# Pipeline for the second model with L1 regularisation
stageslrL1 = [vecAssembler, lrL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(trainingData)

predictions = pipelineModellrL1.transform(testData)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

# 保存 vector
w_L1 = pipelineModellrL1.stages[-1].coefficients.values


# 对于无正则化情况和 L1 正则化情况, 绘制系数 w 的值
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(w_no_reg)
ax1.set_title('No regularisation')
ax2.plot(w_L1)
ax2.set_title('L1 regularisation')
plt.savefig("Output/w_with_and_without_reg.png")

# 看看每种方法首选哪些功能
#---
# 对于无正则化,最相关的特征为:
spam_names[np.argmax(np.abs(w_no_reg))]
#输出 'word_freq_cs'

# 对于 L1 正则化，最相关的特征是
spam_names[np.argmax(np.abs(w_L1))]
# 输出 'char_freq_$'

# 使用summary方法
lrModel1 = pipelineModellrL1.stages[-1]
lrModel1.summary.accuracy
#输出0.9111922,其值与之前得到的不同

# 其他数据,例如:标签的预测
print("Precision by label:")
for i, prec in enumerate(lrModel1.summary.precisionByLabel):
    print("label %d: %s" % (i, prec))



#---

# part3. 练习
# 练习 1
# 在上述相同的数据分区上进行纯 L2 正则化和弹性网络正则化
# 比较准确性并找到与这两种情况最相关的特征
# 这些特征与 L1 正则化获得的特征相同吗？




# 练习 2
# 创建一个ParamGridBuilder在CrossValidator中使用,
# 用来微调得到最佳正则化类型,和该类型正则化的最佳参数
# 对CrossValidator使用五个folds。
