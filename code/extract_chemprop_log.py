import re 
import glob
from pathlib import Path
import pandas as pd
import argparse
import os

def extract_chemprop_log(base_path, split_type, dataset_name):
    regression = ['delaney','freesolv','lipo']
    if split_type in ['random', 'scaffold', 'umap']:
        log_files = sorted(glob.glob(f'{base_path}/chemprop_train/{split_type}/{dataset_name}/train_{dataset_name}_{split_type}_*.log'))
    elif split_type in ['spectra_tanimoto']:
        log_files = sorted(glob.glob(f'{base_path}/chemprop_train/{split_type}/{dataset_name}/train_{dataset_name}_{split_type}_SP_*.log'))
    rows = []
    for log_file in log_files:
        if split_type in ['random', 'scaffold', 'umap']:
            name = Path(log_file).stem[6:-4]
        elif split_type in ['spectra_tanimoto']:
            name = Path(log_file).stem[-6:]

        with open(log_file, "r") as f:
            text = f.read()

        size_match = re.search(r"train/val/test split_\d+ sizes: \[(\d+), (\d+), (\d+)\]",text)
        if size_match is None:
            print("Could not find splits size")
            continue
        else:
            size = [int(x) for x in size_match.groups()]

        if dataset_name not in regression:
            auc_match = re.findall(r"test/roc:\s*([0-9.]+)", text)
            print(auc_match)
            aucs = [float(a) for a in auc_match]
            row = {
                "AUC": name,
                "Size": size,
                "Ensemble 1": aucs[0] if len(aucs) > 0 else None,
                "Ensemble 2": aucs[1] if len(aucs) > 1 else None,
                "Ensemble 3": aucs[2] if len(aucs) > 2 else None,
                "Ensemble 4": aucs[3] if len(aucs) > 3 else None,
                "Ensemble 5": aucs[4] if len(aucs) > 4 else None
            }
            rows.append(row)
        else:
            rmse_match = re.findall(r"test/rmse:\s*([0-9.]+)", text)
            rmses = [float(a) for a in rmse_match]

            row = {
                "RMSE": name,
                "Size": size,
                "Ensemble 1": rmses[0] if len(rmses) > 0 else None,
                "Ensemble 2": rmses[1] if len(rmses) > 1 else None,
                "Ensemble 3": rmses[2] if len(rmses) > 2 else None,
                "Ensemble 4": rmses[3] if len(rmses) > 3 else None,
                "Ensemble 5": rmses[4] if len(rmses) > 4 else None
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(f'{base_path}/metrics/{split_type}/sheet',exist_ok=True)
    df.to_csv(f"{base_path}/metrics/{split_type}/sheet/{split_type}_metrics_{dataset_name}.csv", index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--split_type', type=str, required=True)
    args = parser.parse_args()
    extract_chemprop_log(args.base_path, args.split_type, args.dataset_name)