# ---
# 1. RDD 和共享变量
# RDD 是 Spark 最底层的数据结构，用于存储 分布式数据。
# 它是 不可变（immutable） 的、支持 分区（partitioned），并且具备 容错（fault-tolerant） 机制
# 特性：
#     分布式：RDD 自动分割数据，并在多个节点并行处理。
#     容错性：RDD 通过 Lineage（血统） 追踪数据操作，即使某个分区丢失，仍能恢复数据。
#     懒加载（Lazy Evaluation）：RDD 只有在真正执行 Action 操作（如 count()、collect()）时才会计算，提高效率。
# RDD操作：
#     转换（Transformations）：创建新的 RDD，不会立即执行
#     行动（Actions）：触发计算并返回结果

# 2. DataFrame（Spark SQL DataFrame）
# DataFrame 是 Spark 计算的高级 API，
#     类似于 Pandas 的 DataFrame，但能在 HPC 并行计算 中高效处理大规模数据

# 3. ML Pipeline（机器学习管道）
# ML Pipeline 是 Spark 提供的 自动化建模流程
# 包含：
#     1 数据预处理 2 特征工程 3 模型训练 4 模型评估 5 超参数优化
#
# 4. 并行化（Parallelization）
# HPC 的核心是 并行计算，Spark 通过 RDD、DataFrame 和 ML Pipeline 实现高效分布式计算
# 关键机制：
#     1.数据分区（Partitioning） 2.任务调度（Task Scheduling） 3.惰性计算（Lazy Evaluation）
#
# 总结
#     RDD 适合底层分布式计算（但 API 繁琐）。
#     DataFrame 适合 大规模数据分析，支持 SQL，性能更优。
#     ML Pipeline 适合 机器学习建模，简化流程。
#     并行计算 是 Spark 在 HPC 任务中的核心，让计算更快、更高效！
#
# HPC 任务建议：
#     数据处理：使用 DataFrame
#     机器学习：使用 ML Pipeline
#     大规模计算：合理分区，优化 并行化

# ---

# 1. RDD 和共享变量

# 登录到 Stanage 集群
ssh acw24yz@stanage.shef.ac.uk

# 请求 2 个内核
    #实验室队列
srun --account=default --reservation=com6012-2 --cpus-per-task=2 --time=01:00:00 --pty /bin/bash
    #通用队列
srun --pty --cpus-per-task=2 bash -i

# 启动myspark
    # if 创建启动脚本myspark.sh在实验1
    source myspark.sh #（应当位于根目录）
    # else 手动启动
    module load Java/17.0.4
    module load Anaconda3/2022.05
    source activate myspark

# 启动 PySpark shell
conda install -y numpy # install numpy, to be used in Task 3. This ONLY needs to be done ONCE. NOT every time.
cd com6012/mywork # 切换到我们的目录
pyspark --master local[2] # start pyspark with 2 cores requested above.

#---
# `resilient distributed dataset` (RDD)
#使用SparkContext中的parallelize方法，
#   创建parallelized-collections（RDD）
# 例如，创建data的并行化集合rddData
data = [1, 2, 3, 4, 5]
rddData = sc.parallelize(data)
rddData.collect()

#第二个参数并行化传递给 SparkContext：手动设置分区数
sc.parallelize(data, 16)

# π的估计
from random import random, seed
seed(213)

def inside(p):
    x, y = random(), random()
    return x*x + y*y < 1

NUM_SAMPLES = 10000000
count = sc.parallelize(range(0, NUM_SAMPLES),8).filter(inside).count()
print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))
#此时没有指定种子，因此结果可能不一致


# 共享变量：广播变量与累加器

# 广播变量Broadcast variables
    # 为避免创建大型变量副本，而在每台计算机保留的可访问（只读）变量
    # 以序列化形式缓存，并在运行每个任务之前进行反序列化
# 通过`SparkContext.broadcast(v)`创建广播变量“$v$”
# 通过.value访问v的值
broadcastVar = sc.broadcast([1, 2, 3])
broadcastVar #输出广播变量属性
broadcastVar.value #输出广播变量值

# 累加器Accumulators
    # 仅通过结合和交换作“添加”的变量，因此可以有效地并行支持
# 通过`SparkContext.accumulator(v)`创建累加器初始值"v"
# Cluster tasks随后可以通过`add`方法添加，但其无法读取值
# 只有驱动能使用`value`方法读取累加器的值
accum = sc.accumulator(0)
accum
# Accumulator<id=0, value=0>
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value
# 10
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value
# 20


#---
# 2. DataFrame
# 引入SparkSession后，RDD已被替换为dataset，其可用于：
#     转换 transformations：生成新数据集
#     actions：返回结果的计算
# 创建 DataFrame 和 数据集：
rdd = sc.parallelize([(1,2,3),(4,5,6),(7,8,9)])
df = rdd.toDF(["a","b","c"])
#测试输出
rdd

#看看 DataFrame
df
df.show()
df.printSchema()

#从 DataFrame 获取 RDD
rdd2=df.rdd
rdd2
rdd2.collect()  # view the content

#从 CSV 文件加载数据
df = spark.read.load("Data/Advertising.csv", format="csv", inferSchema="true", header="true")
df.show(5)  # show the top 5 rows

#CSV 文件是半结构化数据，因此spark自动推断了方案。
df.printSchema()
# 删除第一列
df2=df.drop('_c0')
df2.printSchema()

#使用 .describe（）.show（） 获取数字列的摘要统计数据
df2.describe().show()