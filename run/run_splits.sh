#!/bin/bash

#SBATCH --job-name=spectra_splits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --output=slurm_logs/splits/%A_%a.out
#SBATCH --error=slurm_logs/splits/%A_%a.err

PROJECT_DIR="/home/vndo_umass_edu"

DATASETS=(clintox delaney)
CURRENT_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "-Start SLURM Job-"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Project directory: $PROJECT_DIR"

mkdir -p $PROJECT_DIR/slurm_logs/splits

echo "Loading Conda environment"
module purge
module load conda/latest
conda activate molml

echo "rdkit version"
python -c "from rdkit import rdBase; print(rdBase.rdkitVersion)"

cd $PROJECT_DIR
echo "Running random, scaffold, umap splits"
python -u code/splits.py \
        --dataset_name $CURRENT_DATASET \
        --base_path $PROJECT_DIR

echo "Finished $CURRENT_DATASET"