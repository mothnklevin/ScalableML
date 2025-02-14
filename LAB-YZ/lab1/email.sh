#!/bin/bash
#SBATCH --job-name=Test_Email
#SBATCH --time=02:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=./Output/test_email.txt
#SBATCH --mail-user=yzhang851@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Test email notification"

