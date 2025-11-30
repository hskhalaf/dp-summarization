#!/bin/bash
#SBATCH -c 2 
#SBATCH -p seas_gpu
#SBATCH --mem=80000
#SBATCH -t 0-6:00:00
#SBATCH -o dp.out
#SBATCH -e dp.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

source ~/.bashrc
conda activate llmenv
export PYTHONUNBUFFERED=1
python make_summaries.py 
