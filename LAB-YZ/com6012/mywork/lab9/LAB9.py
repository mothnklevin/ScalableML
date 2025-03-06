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

neg
# LabeledPoint(0.0, (3,[0,2],[1.0,3.0]))
neg.label
# 0.0
neg.features
# SparseVector(3, {0: 1.0, 2: 3.0})

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


