#!/bin/bash

#SBATCH --job-name=chemprop_splits_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --array=0-35
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_data/%A_%a.out
#SBATCH --error=slurm_logs/chemprop_data/%A_%a.err

PROJECT_DIR="/home/vndo_umass_edu"

DATASETS=(bace bbbp clintox delaney freesolv hiv lipo sider tox21)
SPLITS=(random scaffold umap spectra_tanimoto)

NUM_DATASETS=${#DATASETS[@]}
NUM_SPLITS=${#SPLITS[@]}

IDX=$((SLURM_ARRAY_TASK_ID - 1))

dataset_idx=$(( IDX % NUM_DATASETS ))
split_idx=$(( IDX / NUM_DATASETS ))

CURRENT_DATASET=${DATASETS[$dataset_idx]}
CURRENT_SPLIT=${SPLITS[$split_idx]}

echo "-Start SLURM Job-"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Project directory: $PROJECT_DIR"

mkdir -p $PROJECT_DIR/slurm_logs/chemprop_data

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

cd $PROJECT_DIR

echo "Running chemprop_data to get splits data in CSV"
python -u code/chemprop_data.py \
        --dataset_name $CURRENT_DATASET \
        --base_path $PROJECT_DIR \
        --split_type $CURRENT_SPLIT

echo "Finished $CURRENT_DATASET"