#!/bin/bash

#SBATCH --job-name=chemprop_hpopt
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=2
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_hpopt/%x_%A_%a.out
#SBATCH --error=slurm_logs/chemprop_hpopt/%x_%A_%a.err

project_dir=/home/vndo_umass_edu

datasets=(bace bbbp clintox sider)
current_dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running $current_dataset"

results_dir=$project_dir/chemprop_hpopt/$current_dataset
data_path=$project_dir/splits_data/hpopt/$current_dataset/data.csv
splits_path=$project_dir/splits_data/hpopt/$current_dataset/data.json
RAY_TEMP_DIR=$project_dir/ray/ray_temp

mkdir -p slurm_logs/chemprop_hpopt/$current_dataset
mkdir -p $results_dir
mkdir -p $RAY_TEMP_DIR

log_dir=slurm_logs/chemprop_hpopt/$current_dataset
mkdir -p $log_dir

mv slurm_logs/chemprop_hpopt/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $log_dir
mv slurm_logs/chemprop_hpopt/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $log_dir

echo "Loading Conda environment"
module purge
module load conda/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate molml
python --version

echo "Run hyperparameter search"

chemprop hpopt \
-t classification \
--data-path $data_path \
--splits-file $splits_path \
--num-workers 20 \
--raytune-num-samples 100 \
--epochs 200 \
--ensemble-size 0 \
--aggregation norm \
--raytune-num-cpus 40 \
--raytune-num-gpus 2 \
--raytune-max-concurrent-trials 2 \
--search-parameter-keywords depth ffn_num_layers message_hidden_dim ffn_hidden_dim dropout batch_size init_lr_ratio max_lr final_lr_ratio warmup_epochs \
--hyperopt-random-state-seed 42 \
--hpopt-save-dir $results_dir

echo "Done $current_dataset"