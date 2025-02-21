# part1. PySpark 中的 GLM
rawdata = spark.read.csv('./assets/hour.csv', header=True)
rawdata.cache()

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])

# 转换为 DoubleType
new_schemaNames = new_rawdata.schema.names
from pyspark.sql.types import DoubleType
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))
# 打印检查数据
new_rawdata.printSchema()

# 创建训练和测试数据
(trainingData, testData) = new_rawdata.randomSplit([0.7, 0.3], 42)
# 检查格式
new_schemaNames[0:new_ncolumns-3]

# 将特征组装成一个向量
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = new_schemaNames[0:new_ncolumns-3], outputCol = 'features')

# 对数据集应用 Poisson 回归。
# 使用 GeneralizedLinearRegression 模型
from pyspark.ml.regression import GeneralizedLinearRegression
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
# 创建pipeline
from pyspark.ml import Pipeline
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)

# 将管道拟合到数据集
pipelineModel = pipeline.fit(trainingData)

# 评估 RMSE
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)

# 检查向量
pipelineModel.stages[-1].coefficients



#---
# part2. 练习
# 练习1
# 变量 season、yr、mnth、hr、holiday、weekday、workingday 和 weathersit 是被视为连续变量的分类变量。
# 一般来说，这不是最佳选择，因为我们间接地将几何或顺序施加到不需要具有此类几何的变量上。
# 例如，变量 season 采用值 1 （春季）、2 （夏季）、3 （秋季） 和 4 （冬季）。
# 间接地，我们说春天和冬天之间的距离（1 和 4）大于春天 （1） 和夏天 （3） 之间的距离。
# 这其实没有原因。
# 为了避免这种强加的几何图形在不遵循 1 的变量上，
# 通常的方法是将分类特征转换为 one-hot 编码的表示形式。
#
# 对 Bike Sharing 数据集使用 OneHotEncoder 估计器来表示分类变量。
# 使用相同的训练和测试数据，
# 使用相同的 Poisson 模型计算测试数据的 RMSE。


# 练习2
# 使用以下算法比较线性回归在相同数据集上的性能：
#
# 线性回归使用ℓ1正则化和 OWL-QN优化。
# 线性回归使用elastic Net正则化和OWL-QN优化。
# 线性回归使用ℓ2正则化和L-BGFS优化。
# 线性回归使用ℓ2正则化和IRLS优化。