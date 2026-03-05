import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os
import json
from pathlib import Path
import pickle 
from os.path import join
import argparse
import random

def hpopt_random_split(base_path, dataset_name):
    df = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
    mfp = []
    for i in range(len(df['smiles'])):
        mol = Chem.MolFromSmiles(df['smiles'][i])
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        mfp.append(fp)
    dataset = dc.data.NumpyDataset(y=df.drop(columns='smiles').values, ids=df['smiles'].values, X=mfp)

    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                                         seed=42)
    idx_train, idx_valid, idx_test = len(train) - 1, len(train) - 1 + len(valid) - 1, len(dataset) - 1

    save_dir = f'{base_path}/splits_data/hpopt/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    data_dic = [
        {
            'train': f"0 - {idx_train}",
            'val': f"{idx_train+1} - {idx_valid}",
            'test': f"{idx_valid+1} - {idx_test}"
        }
    ]
    json_str = json.dumps(data_dic, indent=3)
    with open(f'{save_dir}/data.json', "w") as f:
        f.write(json_str)

    concat_df = pd.DataFrame()
    splits = {'train': train, 'val': valid, 'test': test}
    for name, obj in splits.items():
        file_path = f'{save_dir}/{name}.csv'
        data = df.loc[df['smiles'].isin(obj.ids)]
        data.to_csv(file_path, index=False)
        concat_df = pd.concat([concat_df, data])
    concat_df.to_csv(f'{save_dir}/data.csv', index=False)

    return None

def chemprop_data(base_path, dataset_name, split_type):
    df = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
    save_dir = f'{base_path}/splits_data/chemprop_data/{split_type}/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    if split_type in ['spectra_tanimoto']:
        print(f'Starting with {dataset_name}_{split_type}')
        root = Path(base_path) / "splits" / split_type / f"{dataset_name}_SPECTRA_splits"
        for parameter in range(21):
            parameter = f'{parameter/20:.2f}'
            save_sp_dir = f'{save_dir}/SP_{parameter}'
            os.makedirs(save_sp_dir,exist_ok=True)
            for i in range(3):
                sp_dir = root / f"SP_{parameter}_{i}"
                train_file = sp_dir / "train.pkl"
                test_file = sp_dir / "test.pkl"
                stats_file = sp_dir / "stats.pkl"

                if not (train_file.exists() and test_file.exists() and stats_file.exists()):
                    continue
                with train_file.open("rb") as f:
                    train_indices = pickle.load(f)
                with test_file.open("rb") as f:
                    test_indices =pickle.load(f)
                val_indices = np.random.choice(train_indices, len(train_indices) // 8, replace=False)
                val_indices = [int(x) for x in val_indices]
                train_indices = [int(idx) for idx in train_indices if idx not in val_indices]
                test_indices = [int(idx) for idx in test_indices]
                data_dic = [
                            {
                                'train': train_indices,
                                'val': val_indices,
                                'test': test_indices,
                            }
                            ]
                json_str = json.dumps(data_dic)
                with open(f'{save_sp_dir}/data_{i}.json', "w") as f:
                    f.write(json_str)
        print(f'Finish {dataset_name}_{split_type}')
    else:
        print(f'Starting with {dataset_name}_{split_type}')
        for i in range(5):
            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_train_split_{i}.pkl'),
                      'rb') as f:
                train_indices = pickle.load(f)
            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_test_split_{i}.pkl'),
                      'rb') as f:
                test_indices = pickle.load(f)
            val_indices = np.random.choice(train_indices, len(train_indices) // 8, replace=False)
            val_indices = val_indices.tolist()
            train_indices = [idx for idx in train_indices if idx not in val_indices]
            data_dic = [
                            {
                                'train': train_indices,
                                'val': val_indices,
                                'test': test_indices
                            }
                            ]
            json_str = json.dumps(data_dic)
            with open(f'{save_dir}/data_{i}.json', "w") as f:
                f.write(json_str)
        print(f'Finish {dataset_name}_{split_type}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create Splits CSV for Chemprop hyperparameter search and retraining')
    parser.add_argument('--dataset_name', type =str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--split_type', type=str, required=False)
    args = parser.parse_args()
    chemprop_data(args.base_path, args.dataset_name, args.split_type)
    