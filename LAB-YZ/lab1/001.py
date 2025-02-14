# 1 安装与环境配置
#进入登录节点：
srun --account=default --reservation=com6012-1 --time=00:30:00 --pty /bin/bash # 替换-1为-【实验室ID】，只能在实验室时间访问实验室节点
srun --pty bash -i # 进入普通节点
#加载 Java 和 conda
module load Java/17.0.4
module load Anaconda3/2022.05
#创建myspark虚拟环境，不要更新conda
conda create -n myspark python=3.11.7
# 激活环境。激活后，应在前面看到（myspark）
source activate myspark
# 使用pip，继续选y
pip install pyspark==3.5.0

# 重置环境
# 1 注销 Stanage 并重新登录，然后启动交互式会话 。run --pty bash -i
# 2 通过命令 resetenv 恢复默认文件。如果命令resetenv不起作用（可能已从 $PATH 中删除），您可以直接运行 resetenv 命令：/opt/site/bin/resetenv。
# 3 通过 `rm -rf ~/.conda`和`rm -rf ~/.condarc` 删除 conda 环境文件。
# 4 完全退出然后再次登录。
# 5 启动交互式会话并按照上述步骤创建myspark环境。

# 运行 pyspark
pyspark
# 现在已进入pyspark。使用 Ctrl + D 退出shell

#如果在进入`pyspark` shell时遇到`segmentation fault`问题，
#运行`export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8`修复

# 传输文件：
FileZilla


# 2 运行myspark
# 获取节点并激活 myspark：
`srun --account=default --reservation=com6012-$LAB_ID --time=00:30:00 --pty /bin/bash`
`srun --pty bash -i`
# 激活环境
module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark
    # 或者：
    # put `HPC/myspark.sh`在”root“目录下 ，并依次运行上面3个命令
# 运行 pyspark
pyspark  # pyspark --master local[4] for 4 cores
# 检查您的 SparkSession 和 SparkContext 对象
# 应类似于：
# >>> spark
# <pyspark.sql.session.SparkSession object at 0x7f82156b1750>
# >>> sc
# <SparkContext master=local[*] appName=PySparkShell>

# 3 使用 Spark 进行日志挖掘 示例
# 对于file not foundcannot ，open file：
# 首先，确保文件位于正确的目录中，并在必要时更改文件路径
# 通过 Ctrl + D 退出 pyspark 。看看当前位置
# 输出应为
# (myspark) [abc1de@node*** [stanage] ~]$ pwd
# /users/abc1de
# 创建一个名为 com6012 的新目录并转到它
mkdir com6012
cd com6012
# 复制我们的教材
git clone --depth 1 https://github.com/COM6012/ScalableML
# 如果`ScalableML` is not empty：
`rm -rf ScalableML`
#建议在`com6012`下，为作业建立单独文件夹`mywork`
# 随后检查：
ls
cd ScalableML
ls
pwd

# 随后重新启动shell
pyspark
#查看Data中的日志文件`NASA_Aug95_100.txt`
logFile=spark.read.text("Data/NASA_Aug95_100.txt")
# 检查文件
logFile
# 计算行数
logFile.count()
# 查看第一行
logFile.first()

# 示例问题：有多少次来自日本的访问？
# 解答步骤：
# 查找来自日本的日志
hostsJapan = logFile.filter(logFile.value.contains(".jp"))
# 先检查5个日志，以确定获取的内容正确
hostsJapan.show(5,False)
# 确认正确后进行计数
hostsJapan.count()

# 4 独立应用程序
#主目录：/users/$USER ，可以使用~切换到目录
# 快速存储目录：/mnt/parscratch/

# 在`/mnt/parscratch/users`建立个人文件夹
# mkdir -m 0700 /mnt/parscratch/users/YOUR_USERNAME
mkdir -m 0700 /mnt/parscratch/users/acw24yz
#创造`LogMining100.py`文件，实际上已包含在code文件夹中
#提交文件并运行
spark-submit Code/LogMining100.py



# 查看文件：
# 使用less命令（分页查看，适合大文件）
less Output/COM6012_Lab1.txt
# 使用 ↑ / ↓ 或 PageUp / PageDown 翻页。
# 按 q 退出查看模式。
# 只查看前几行
head -n 10 Output/COM6012_Lab1.txt
# -n 10 代表查看前 10 行



#实验：Lab1_NASA_logAnalysis.sh

#!/bin/bash
#SBATCH --job-name=NASA_Log_Analysis

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --output=./Output/COM6012_Lab1_NASA.txt  # 任务输出日志
#SBATCH --mail-user=yzhang851@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

#交互式处理
spark-submit ./LogAnalysis01.py



#修改spark-submit ./LogMiningBig.py 中的py文件为现task代码
#例：spark-submit ./LogAnalysis.py
#创建 LogAnalysis.py
nano LogAnalysis01.py
#写入代码
from pyspark.sql import SparkSession
# 创建 SparkSession
spark = SparkSession.builder \
    .appName("NASA Log Analysis") \
    .config("spark.local.dir", "/mnt/parscratch/users/acw24yz") \
    .getOrCreate()
# 读取 NASA 日志数据
logFile = spark.read.text("NASA_access_log_Aug95.gz").cache()
# 1. 总请求数
totalRequests = logFile.count()
# 2. 来自 gateway.timken.com 的请求数
requestsFromTimken = logFile.filter(logFile.value.contains("gateway.timken.com")).count()
# 3. 1995年8月15日的请求数
requestsOnAug15 = logFile.filter(logFile.value.contains("15/Aug/1995")).count()
# 4. 总 404 错误数
errors404Total = logFile.filter(logFile.value.contains(" 404 ")).count()
# 5. 1995年8月15日的 404 错误数
errors404OnAug15 = logFile.filter((logFile.value.contains("15/Aug/1995")) & (logFile.value.contains(" 404 "))).count()
# 6. 来自 gateway.timken.com 的 404 错误数（1995年8月15日）
errors404FromTimkenOnAug15 = logFile.filter((logFile.value.contains("15/Aug/1995")) &
                                            (logFile.value.contains(" 404 ")) &
                                            (logFile.value.contains("gateway.timken.com"))).count()
# 打印结果
print("\nNASA Log Analysis Results:")
print(f"1. Total requests: {totalRequests}")
print(f"2. Requests from gateway.timken.com: {requestsFromTimken}")
print(f"3. Requests on 15th August 1995: {requestsOnAug15}")
print(f"4. Total 404 errors: {errors404Total}")
print(f"5. 404 errors on 15th August: {errors404OnAug15}")
print(f"6. 404 errors from gateway.timken.com on 15th August: {errors404FromTimkenOnAug15}\n")

# 关闭 SparkSession
spark.stop()


# 开始批处理
sbatch HPC/Lab1_NASA_logAnalysis.sh
    # 提交后记住id
    Submitted batch job 5407310
# 检查任务状态
squeue -u acw24yz
# 如果已完成，可以查看结果
cat Output/COM6012_Lab1_NASA.txt # 直接输出所有内容
less Output/COM6012_Lab1_NASA.txt # 分页显示

#最后结果

NASA Log Analysis Results:
# 1. Total requests: 1569898
# 2. Requests from gateway.timken.com: 35
# 3. Requests on 15th August 1995: 58847
# 4. Total 404 errors: 10056
# 5. 404 errors on 15th August: 327
# 6. 404 errors from gateway.timken.com on 15th August: 3
