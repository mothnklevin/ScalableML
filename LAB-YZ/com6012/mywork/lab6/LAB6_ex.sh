#!/bin/bash
#SBATCH --job-name=LAB5_EX

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --output=/users/acw24yz/com6012/mywork/Output/LAB6.txt  # task output log
#SBATCH --mail-user=yzhang851@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

#spark-submit --driver-memory 5g --executor-memory 5g --master local[2] ./test.py
#spark-submit ./LAB3_01_BaseMD.py
spark-submit --driver-memory 6g --executor-memory 5g --master local[4] ./LAB6_01.py
#spark-submit --driver-memory 6g --executor-memory 5g --master local[4] ./LAB6_02.py


#grep -v -E "INFO|rdd_[0-9]"  \
grep -v -E "INFO|CrossValidator_|Project|Sort|Sample" \
    /users/acw24yz/com6012/mywork/Output/LAB6.txt > \
    /users/acw24yz/com6012/mywork/Output/LAB6_clean.txt


echo "LAB6  completed successfully"
