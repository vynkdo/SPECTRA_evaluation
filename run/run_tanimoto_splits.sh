#!/bin/bash

#SBATCH --job-name=spectra_splits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --array=0-6
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/spectra_splits/%A_%a.out
#SBATCH --error=slurm_logs/spectra_splits%A_%a.err

PROJECT_DIR="/home/vndo_umass_edu"

DATASETS=(bace bbbp clintox delaney freesolv lipo sider)
CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "-Start SLURM Job-"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Project directory: $PROJECT_DIR"

mkdir -p $PROJECT_DIR/slurm_logs/spectra_splits

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

cd $PROJECT_DIR

echo "Running spectra-splits"
python -u code/spectra_splits.py \
        --dataset_name $CURRENT_DATASET \
        --base_path $PROJECT_DIR

echo "Finished $CURRENT_DATASET"