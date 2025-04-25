#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00               # Adjust time if needed
#SBATCH --gpus=h100-80:1
#SBATCH --array=0-0             # Only 2 tasks needed (0 and 1)
#SBATCH -o output_%A_%a.out      # Output file per task (%A=job ID, %a=array index)
#SBATCH -e error_%A_%a.err       # Error file per task (%A=job ID, %a=array index)

module load anaconda3
conda activate neuralode # Or your correct conda environment name
cd "/jet/home/azhang19/stat 214/stat-214-lab3-group6" # Change to the project directory

# --- Run the Python script ---
# Use the variables to set the command-line arguments
python -u "/jet/home/azhang19/stat 214/stat-214-lab3-group6/code/BERT/train.py" \
    --dim 64 \
    --hidden-dim 224 \
    --mlm-prob 0.2 \
    --save-path "/jet/home/azhang19/stat 214/stat-214-lab3-group6/code/ckpts"

echo "Task $SLURM_ARRAY_TASK_ID finished."