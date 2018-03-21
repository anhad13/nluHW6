#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1

python script.py