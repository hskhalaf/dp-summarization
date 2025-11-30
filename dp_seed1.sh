#!/bin/bash
#SBATCH -c 2 
#SBATCH -p seas_gpu
#SBATCH --mem=80000
#SBATCH -t 0-18:00:00
#SBATCH -o dp_seed1_%j.out
#SBATCH -e dp_seed1_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

source ~/.bashrc
conda activate llmenv
export PYTHONUNBUFFERED=1

seed=1
temps=(1.0)
epsilon_min=10
epsilon_max=120
epsilon_steps=20
num_products=100

mkdir -p results

for temp in "${temps[@]}"; do
    output_file="results/results_seed${seed}_temp${temp}.txt"
    echo "====================================="
    echo "Running: seed=$seed, temp=$temp"
    echo "Output: $output_file"
    echo "====================================="
    
    python make_summaries.py \
        --seed "$seed" \
        --temperature "$temp" \
        --num_products "$num_products" \
        --epsilon_min "$epsilon_min" \
        --epsilon_max "$epsilon_max" \
        --epsilon_steps "$epsilon_steps" \
        --output "$output_file"
    
    echo "Completed: $output_file"
    echo ""
done

echo "Seed 1 all runs completed!"
