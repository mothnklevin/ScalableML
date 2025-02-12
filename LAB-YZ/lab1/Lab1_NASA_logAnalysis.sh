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

spark-submit ./LogAnalysis01.py