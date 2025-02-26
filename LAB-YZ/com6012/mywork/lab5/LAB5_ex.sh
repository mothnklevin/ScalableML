#!/bin/bash
#SBATCH --job-name=LAB5_EX

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --output=/users/acw24yz/com6012/mywork/Output/Lab4.txt  # task output log
#SBATCH --mail-user=yzhang851@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

#spark-submit --driver-memory 5g --executor-memory 5g --master local[2] ./test.py
#spark-submit ./LAB3_01_BaseMD.py
spark-submit --driver-memory 4g --executor-memory 5g --master local[2] ./LAB5_01.py
spark-submit --driver-memory 4g --executor-memory 5g --master local[2] ./LAB5_02.py
spark-submit --driver-memory 4g --executor-memory 5g --master local[2] ./LAB5_03.py



grep -v "INFO"  /users/acw24yz/com6012/mywork/Output/Lab5.txt > \
                /users/acw24yz/com6012/mywork/Output/Lab5_clean.txt

echo "LAB4  completed successfully"
