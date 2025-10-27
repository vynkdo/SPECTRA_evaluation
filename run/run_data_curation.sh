#!/bin/bash

#SBATCH --job-name=data_curation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

PROJECT_DIR="/home/vndo_umass_edu/dataset"

DATASETS=(bace bbbp clintox hiv sider tox21)
CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "-Start SLURM Job-"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Project directory: $PROJECT_DIR"

mkdir -p $PROJECT_DIR/slurm_logs

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

cd $PROJECT_DIR

echo "Running data_curation"
python -u data_curation.py \
        --dataset_name $CURRENT_DATASET \
        --base_path $PROJECT_DIR \
        --task 'classification' 

echo "Finished $CURRENT_DATASET"