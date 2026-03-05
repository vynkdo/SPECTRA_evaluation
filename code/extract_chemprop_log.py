import re #for searching
import glob
from pathlib import Path
import pandas as pd
import argparse
import os

def extract_chemprop_log(path_to_log,dataset_name):
    regression = ['delaney','freesolv','lipo']
    log_files = sorted(glob.glob(f'{path_to_log}/{dataset_name}/train_{dataset_name}_spectra_tanimoto_SP_*.log'))
    rows = []
    for log_file in log_files:
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
    os.makedirs(f'{path_to_log}/sheet',exist_ok=True)
    df.to_excel(f"{path_to_log}/sheet/spectra_ensemble_{dataset_name}.xlsx", index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--path_to_log', type=str, required=True)
    args = parser.parse_args()
    extract_chemprop_log(args.path_to_log, args.dataset_name)