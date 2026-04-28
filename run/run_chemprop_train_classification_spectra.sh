#!/bin/bash

#SBATCH --job-name=chemprop_train_spectra_classification
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --array=1-315
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/chemprop_train_spectra/%x_%A_%a.out
#SBATCH --error=slurm_logs/chemprop_train_spectra/%x_%A_%a.err

PROJECT_DIR=/home/vndo_umass_edu/spectra

DATASETS=(bace bbbp clintox sider tox21)
SP=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
NUMS=(0 1 2)

NUM_DATASETS=${#DATASETS[@]}
NUM_SP=${#SP[@]}
NUM_NUMS=${#NUMS[@]}

TOTAL_JOBS=$((NUM_DATASETS * NUM_SP * NUM_NUMS))
IDX=$((SLURM_ARRAY_TASK_ID - 1))

dataset_idx=$(( IDX % NUM_DATASETS ))
sp_idx=$(( (IDX / NUM_DATASETS) % NUM_SP ))
num_idx=$(( (IDX / (NUM_DATASETS * NUM_SP)) % NUM_NUMS ))

CURRENT_DATASET=${DATASETS[$dataset_idx]}
CURRENT_SP=${SP[$sp_idx]}
CURRENT_NUM=${NUMS[$num_idx]}

echo "Running $CURRENT_DATASET"

BEST_CONFIG_DIR=$PROJECT_DIR/chemprop_hpopt/$CURRENT_DATASET
RESULTS_DIR=$PROJECT_DIR/chemprop_train/spectra_tanimoto/$CURRENT_DATASET
DATA_PATH=$PROJECT_DIR/dataset/curated_dataset/$CURRENT_DATASET.csv
SPLITS_PATH=$PROJECT_DIR/splits_data/chemprop_data/spectra_tanimoto/$CURRENT_DATASET/SP_$CURRENT_SP/data_$CURRENT_NUM.json

LOG_DIR=slurm_logs/chemprop_train_spectra/$CURRENT_DATASET
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

mv slurm_logs/chemprop_train_spectra/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out $LOG_DIR
mv slurm_logs/chemprop_train_spectra/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err $LOG_DIR

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

echo "Run chemprop retrain on $CURRENT_DATASET in spectra_tanimoto"

chemprop train \
--log $RESULTS_DIR/train_${CURRENT_DATASET}_spectra_tanimoto_SP_${CURRENT_SP}_${CURRENT_NUM}.log \
-t classification \
--config-path $BEST_CONFIG_DIR/best_config.toml \
--data-path $DATA_PATH \
--splits-file $SPLITS_PATH \
--num-workers 20 \
--epochs 200 \
--patience 15 \
--pytorch-seed 42 \
--aggregation norm \
--ensemble-size 5 \
--metrics roc \
--save-dir $RESULTS_DIR

echo "Done $CURRENT_DATASET"
