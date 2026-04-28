import numpy as np
from os.path import join
from pathlib import Path
import json
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
import argparse


def convert_to_morgan_fingerprint(dataset_name, base_path):
  dataset = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
  dataset_smiles = dataset['smiles']

  mfp = []
  for i in range(len(dataset_smiles)):
    mol = Chem.MolFromSmiles(dataset_smiles[i])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    mfp.append(fp)
  return mfp

def compute_cso(train, test_mfp):
    similarity = []
    for train_mfp in train:
        similarity.extend(BulkTanimotoSimilarity(train_mfp, test_mfp))
    return np.mean(similarity)

def cross_split_overlap(dataset_name, split_type, base_path):
    mfp = convert_to_morgan_fingerprint(dataset_name, base_path)
    rows = []
    if split_type in ['random', 'scaffold', 'umap']:
        for i in range(5):
            with open(join(base_path, f'splits_data/chemprop_data/{split_type}/{dataset_name}/data_{i}.json'), 'r') as f:
                data = json.load(f)
            splits = data[0]
            train_indices = splits['train']
            val_indices = splits['val']
            test_indices = splits['test']
            
            train = [mfp[m] for m in train_indices]
            test = [mfp[n] for n in test_indices]

            cross_split_overlap = compute_cso(train, test)

            rows.append({'index': f'{dataset_name}_{split_type}_{i}',
                        'train_size': len(train),
                        'val_size': len(val_indices),
                        'test_size': len(test),
                        'cross_split_overlap': cross_split_overlap})
    elif split_type in ['spectra_tanimoto']:
        for parameter in range(21):
            parameter = f'{parameter/20:.2f}'
            for i in range(3):
                file_path = Path(join(base_path, f'splits_data/chemprop_data/{split_type}/{dataset_name}/SP_{parameter}/data_{i}.json'))
                if not file_path.exists():
                    continue
                with open(file_path, 'r') as f:
                    data = json.load(f)
                splits = data[0]
                train_indices = splits['train']
                val_indices = splits['val']
                test_indices = splits['test']
                
                train = [mfp[m] for m in train_indices]
                test = [mfp[n] for n in test_indices]

                cross_split_overlap = compute_cso(train, test)
                rows.append({'index': f'{dataset_name}_{split_type}_SP_{parameter}_{i}',
                            'train_size': len(train),
                            'val_size': len(val_indices),
                            'test_size': len(test),
                            'SPECTRA_parameter': parameter,
                            'cross_split_overlap': cross_split_overlap})

    df = pd.DataFrame(rows).set_index('index')
    csv_path = join(base_path, f'splits_data/cross_split_overlap/{split_type}/{dataset_name}_{split_type}_cross_split_overlap.csv')
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    return print(f'Done with {dataset_name}_{split_type}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compute cross-split overlap on train and test sets')
    parser.add_argument('--dataset_name', type =str, required=True)
    parser.add_argument('--split_type', type =str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    args = parser.parse_args()
    cross_split_overlap(args.dataset_name, args.split_type, args.base_path)
