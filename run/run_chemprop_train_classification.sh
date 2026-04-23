#!/bin/bash

#SBATCH --job-name=chemprop_train
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=2
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --array=1-21
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_train/%x_%A_%a.out
#SBATCH --error=slurm_logs/chemprop_train/%x_%A_%a.err

project_dir=/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies

DATASETS=(clintox)
SPLITS=(random)
NUMS=(0 1 2)

NUM_DATASETS=${#DATASETS[@]}
NUM_SPLITS=${#SPLITS[@]}
NUM_NUMS=${#NUMS[@]}

TOTAL_JOBS=$((NUM_DATASETS * NUM_SPLITS * NUM_NUMS))
IDX=$((SLURM_ARRAY_TASK_ID - 1))

dataset_idx=$(( IDX % NUM_DATASETS ))
split_idx=$(( (IDX / NUM_DATASETS) % NUM_SPLITS ))
num_idx=$(( (IDX / (NUM_DATASETS * NUM_SPLITS)) % NUM_NUMS ))

CURRENT_DATASET=${DATASETS[$dataset_idx]}
CURRENT_SPLIT=${SPLITS[$split_idx]}
CURRENT_NUM=${NUMS[$num_idx]}

echo "Running $CURRENT_DATASET"

best_config_dir=$project_dir/chemprop_hpopt/$CURRENT_DATASET
results_dir=$project_dir/chemprop_train_test/$CURRENT_SPLIT/$CURRENT_DATASET
data_path=$project_dir/dataset/curated_dataset/$CURRENT_DATASET.csv
splits_path=$project_dir/splits_data/chemprop_data/$CURRENT_SPLIT/$CURRENT_DATASET/$CURRENT_NUM.json
RAY_TEMP_DIR=$project_dir/ray/ray_temp

log_dir=slurm_logs/chemprop_train/$CURRENT_SPLIT/$CURRENT_DATASET
mkdir -p $log_dir
mkdir -p $results_dir
mkdir -p $RAY_TEMP_DIR

mv slurm_logs/chemprop_train/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $log_dir
mv slurm_logs/chemprop_train/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $log_dir

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

echo "Run chemprop retrain on $CURRENT_DATASET in $CURRENT_SPLIT"

chemprop train \
--log $results_dir/test_${CURRENT_DATASET}_${CURRENT_SPLIT}_${CURRENT_NUM}.log \
-t classification \
--config-path $best_config_dir/best_config.toml \
--data-path $data_path \
--splits-file $splits_path \
--num-workers 20 \
--epochs 200 \
--patience 15 \
--pytorch-seed 42 \
--aggregation norm \
--ensemble-size 5 \
--metrics roc \
--save-dir $results_dir

echo "Done $CURRENT_DATASET"
