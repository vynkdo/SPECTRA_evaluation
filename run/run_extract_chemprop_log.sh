#!/bin/bash

#SBATCH --job-name=extract_chemprop_log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_train/%A_%a.out
#SBATCH --error=slurm_logs/chemprop_train/%A_%a.err

PROJECT_DIR="/scratch4/workspace/vndo_umass_edu-simple"
PATH_TO_LOG="/scratch4/workspace/vndo_umass_edu-simple/chemprop_train/spectra_tanimoto"

DATASETS=(bace bbbp clintox sider delaney freesolv lipo tox21)
CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "-Start SLURM Job-"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Project directory: $PROJECT_DIR"

mkdir -p $PROJECT_DIR/slurm_logs/chemprop_train

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

cd $PROJECT_DIR
echo "Running extract_chemprop_log"
python -u code/extract_chemprop_log.py \
        --dataset_name $CURRENT_DATASET \
        --path_to_log $PATH_TO_LOG

echo "Finished $CURRENT_DATASET"