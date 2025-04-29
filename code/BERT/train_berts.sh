#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --array=0-5
#SBATCH -o output_%A_%a.out
#SBATCH -e error_%A_%a.err

module load anaconda3
conda activate neuralode
cd "/jet/home/azhang19/stat 214/stat-214-lab3-group6"

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# --- Define Parameter Arrays ---
# Dimension sets (dim and corresponding hidden_dim)
dims=(32 64)
hidden_dims=(112 224) # Must correspond to dims array

# MLM probabilities
mlm_probs=(0.1 0.15 0.2)

# --- Calculate Parameter Indices from Task ID ---
num_mlm_probs=${#mlm_probs[@]} # Number of MLM probabilities (should be 3)

# Index for dim/hidden_dim pair (changes every num_mlm_probs tasks)
dim_index=$((SLURM_ARRAY_TASK_ID / num_mlm_probs))

# Index for mlm_prob (cycles through 0, 1, 2)
mlm_index=$((SLURM_ARRAY_TASK_ID % num_mlm_probs))

# --- Get Current Parameters ---
current_dim=${dims[dim_index]}
current_hidden_dim=${hidden_dims[dim_index]}
current_mlm_prob=${mlm_probs[mlm_index]}

echo "Running with parameters:"
echo "  --dim $current_dim"
echo "  --hidden-dim $current_hidden_dim"
echo "  --mlm-prob $current_mlm_prob"

# --- Run the Python script ---
python "/jet/home/azhang19/stat 214/stat-214-lab3-group6/code/BERT/train.py" \
    --dim "$current_dim" \
    --hidden-dim "$current_hidden_dim" \
    --mlm-prob "$current_mlm_prob" \
    --save-path "/ocean/projects/mth240012p/azhang19/lab3/ckpts"

echo "Task $SLURM_ARRAY_TASK_ID finished."