#!/bin/bash
#SBATCH --job-name=LAB2_EX

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --output=/users/acw24yz/com6012/mywork/lab2/Output/COM6012_Lab2.txt  # 任务输出日志
#SBATCH --mail-user=yzhang851@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

spark-submit ./LAB2_01_Log.py
spark-submit ./LAB2_02_LR.py
spark-submit ./LAB2_03_ppline.py

grep -v "INFO"  /users/acw24yz/com6012/mywork/lab2/Output/COM6012_Lab2.txt > \
                /users/acw24yz/com6012/mywork/lab2/Output/COM6012_Lab2_clean.txt

echo "LAB2  completed successfully"

"/users/acw24yz/com6012/mywork/lab2/Output/"