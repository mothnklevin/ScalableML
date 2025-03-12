# PART1. 基于 RDD 的 API 中的数据类型

# 1 局部向量：
#   密集向量 (双精度数组) 和 稀疏向量 (两个并行数组：indices 和 values)

import numpy as np
from pyspark.mllib.linalg import Vectors

dv1 = np.array([1.0, 0.0, 3.0])  # Use a NumPy array as a dense vector.
dv2 = [1.0, 0.0, 3.0]  # Use a Python list as a dense vector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])  # Create a SparseVector.

sv1

# 以密集格式查看稀疏向量
sv1.toArray()

# 2 标记点: 与标签/响应关联的局部向量

# 创建一个具有正标签和密集特征向量的标记点 pos
# 以及一个具有负标签和稀疏特征向量的标记点 neg

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

# neg
# # LabeledPoint(0.0, (3,[0,2],[1.0,3.0]))
# neg.label
# # 0.0
# neg.features
# # SparseVector(3, {0: 1.0, 2: 3.0})

# 将特征视为密集向量（而不是稀疏向量）
neg.features.toArray()

# 3 局部矩阵 : 具有 整数类型的行和列索引 以及 双精度类型的值

# 创建一个密集矩阵和一个稀疏矩阵：
from pyspark.mllib.linalg import Matrix, Matrices

dm2 = Matrices.dense(3, 2, [1, 3, 5, 2, 4, 6])
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print(dm2)

print(sm)

# 压缩的稀疏列（CSC 或 CCS）格式用于稀疏矩阵表示
dsm = sm.toDense()

# 分布式矩阵
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mat = RowMatrix(rows)

m = mat.numRows()  # Get its size: m=4, n=3
n = mat.numCols()

rowsRDD = mat.rows  # Get the rows as an RDD of vectors again.

# 以密集矩阵格式查看 RowMatrix
rowsRDD.collect()

# 2. 主成分分析
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
df.show()

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

model.explainedVariance
print(model.pc)



# 基于 RDD 的 API 中的 PCA  pyspark.mllib
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])
rows.collect()

mat = RowMatrix(rows)


pc = mat.computePrincipalComponents(2)
print(pc)

projected = mat.multiply(pc)
projected.rows.collect()

from pyspark.mllib.linalg import DenseVector

denseRows = rows.map(lambda vector: DenseVector(vector.toArray()))
denseRows.collect()

svd = mat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.

print(V)

from pyspark.mllib.feature import StandardScaler

standardizer = StandardScaler(True, False)
model = standardizer.fit(rows)
centeredRows = model.transform(rows)
centeredRows.collect()
# [DenseVector([-2.0, 0.6667, -1.0, 1.3333, -4.0]), DenseVector([0.0, -0.3333, 2.0, -1.6667, 1.0]), DenseVector([2.0, -0.3333, -1.0, 0.3333, 3.0])]
centeredmat = RowMatrix(centeredRows)

svd = centeredmat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.

print(V)

print(s)

evs=s*s
print(evs)

evs/sum(evs)

# # 3. 练习
# 使用 PCA 研究iris.csv数据集
#
# 1. 按照 “使用主成分分析了解降维 （PCA）” 执行相同的分析，
#     从 pyspark.ml 使用基于 DataFrame 的 PCA ：pca.fit()。
# 2. 按照此实验验证：
#     使用其他两个基于 RDD 的 PCA API ：
#     computePrincipalComponents
#     computeSVD
#     是否将提供相同的 PCA 功能

