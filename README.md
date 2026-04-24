# **Chemical space separation: Scaffold and UMAP splits may be closer to Random than we think**

This repository presents a research framework for evaluating chemical data splitting strategies in machine learning and deep learning. We compared state-of-the-art methods including random, scaffold, and UMAP splits by analyzing train–test overlap and its effect on model performance and generalization.

## **List of subdirectories**
- dataset: CSV files containing SMILES strings
- code: code for curating data, generating random, scaffold, UMAP, and SPECTRA splits, calculating cross-split overlap, and organizing splitted data for model training
- splits_data: cross-split ovelap and split indices for hyperparameter optimization and model training
- chemprop_hpopt: best Chemprop configuration for each dataset
- statistical_analyses: code and results for statistical analysis of classical and Chemprop models
- metrics: performance metrics of Chemprop models
- run: Shell files for running jobs on Unity clusters
- plot: figures


## **Installation** 
The required packages and their versions are provided in `requirements.txt`.  

```bash
pip install -r requirements.txt
```

## **Instruction of Running**
### **1. Generate splits**
Data in `dataset` was curated to remove invalid SMILES structure and replicates using `code/data_curation.py`. Next, random, scaffold, and UMAP splits were generated using `code/splits.py` and SPECTRA splits were generated using `code/spectra_splits.py`. Cross-split overlaps of all four splitting strategies was calculated during the execution of `code/spectra_splits.py`. All raw SMILES splits were stored as `.pkl` files and then converted to index-based splits stored as `.json` files using `code/chemprop_data.py`. Final index-based splits were stored in `splits_data/chemprop_data` and `splits_data/hpopt`.  
<br>

### **2. Train classical models**


<br>

### **3. Train Chemprop models**
Chemprop hyperparameter optimization was performed using `run/run_chemprop_hpopt.sh`. The `best_config.toml` files for each dataset stored in `chemprop_hpopt` were then applied to train Chemprop models across all four splitting strategies using `run/run_chemprop_train_classification.sh` and `run/run_chemprop_train_regression.sh`. All metrics were recorded in `metrics` and statistical analysis was performed using `statistical_analyses/stat_significance_chemprop.R`.    

