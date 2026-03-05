#!/bin/bash

#SBATCH --job-name=chemprop_predict
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=2
#SBATCH --mem=20G
#SBATCH --constraint=v100|2080|2080ti|rtx8000|a100|a16|a40|gh200|l40s|l4
#SBATCH --time=1:00:00
#SBATCH --array=1-5
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_predict/%x_%A_%a.out
#SBATCH --error=slurm_logs/chemprop_predict/%x_%A_%a.err

project_dir="/scratch4/workspace/vndo_umass_edu-simple"
DATASETS=(freesolv)
SPLITS=(random)
NUMS=(0 1 2 3 4)

NUM_DATASETS=${#DATASETS[@]}
NUM_SPLITS=${#SPLITS[@]}
NUM_NUMS=${#NUMS[@]}

TOTAL_JOBS=$((NUM_DATASETS * NUM_SPLITS * NUM_NUMS ))
IDX=$((SLURM_ARRAY_TASK_ID - 1))

dataset_idx=$(( IDX % NUM_DATASETS ))
split_idx=$(( (IDX / NUM_DATASETS) % NUM_SPLITS ))
num_idx=$(( (IDX / (NUM_DATASETS * NUM_SPLITS)) % NUM_NUMS ))

CURRENT_DATASET=${DATASETS[$dataset_idx]}
CURRENT_SPLIT=${SPLITS[$split_idx]}
CURRENT_NUM=${NUMS[$num_idx]}

echo "Running $CURRENT_DATASET"

model_dir=$project_dir/chemprop_train/$CURRENT_SPLIT/$CURRENT_DATASET/
result_dir=$project_dir/chemprop_predict/$CURRENT_SPLIT/$CURRENT_DATASET
test_path=$project_dir/splits_data/test_data/$CURRENT_SPLIT/$CURRENT_DATASET/data_$CURRENT_NUM.csv
RAY_TEMP_DIR=$project_dir/ray/ray_temp

log_dir=slurm_logs/chemprop_predict/$CURRENT_SPLIT/$CURRENT_DATASET
mkdir -p $result_dir

mv slurm_logs/chemprop_predict/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $log_dir
mv slurm_logs/chemprop_predict/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $log_dir

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

echo "Run chemprop predict on $CURRENT_DATASET in $CURRENT_SPLIT"

chemprop predict \
--log $result_dir/test_${CURRENT_DATASET}_${CURRENT_SPLIT}_${CURRENT_NUM}.log \
--test-path $test_path \
--output $result_dir/test_prediction_${CURRENT_DATASET}_${CURRENT_SPLIT}_${CURRENT_NUM}.csv \
--model-paths $model_dir