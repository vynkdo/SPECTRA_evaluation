#!/bin/bash

#SBATCH --job-name=chemprop_train
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=2
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_train/%x_%A_%a.out
#SBATCH --error=slurm_logs/chemprop_train/%x_%A_%a.err

project_dir=/home/vndo_umass_edu

datasets=(clintox sider)
current_dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running $current_dataset"

best_config_dir=$project_dir/chemprop_hpopt/$current_dataset
results_dir=$project_dir/chemprop_train/$current_dataset
data_path=$project_dir/splits_data/hpopt/$current_dataset/data.csv
splits_path=$project_dir/splits_data/hpopt/$current_dataset/data.json
RAY_TEMP_DIR=$project_dir/ray/ray_temp

log_dir=slurm_logs/chemprop_train/$current_dataset
mkdir -p $log_dir
mkdir -p $results_dir
mkdir -p $RAY_TEMP_DIR

mv slurm_logs/chemprop_hpopt/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $log_dir
mv slurm_logs/chemprop_hpopt/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $log_dir

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

echo "Run chemprop retrain on $current_dataset"

chemprop train \
-t classification \
--config-path $best_config_dir/best_config.toml \
--data-path $data_path \
--splits-file $splits_path \
--num-workers 20 \
--epochs 200 \
--patience 10 \
--pytorch-seed 42 \
--aggregation norm \
--show-individual-scores \
--ensemble-size 5 \
--metrics roc \
--save-dir $results_dir

echo "Done $current_dataset"
