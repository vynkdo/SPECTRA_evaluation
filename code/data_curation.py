import pandas as pd
from rdkit import Chem
from pathlib import Path
import numpy as np
import os
import argparse

def num_molecule_to_csv(csv_path, dataset_name, number_name:str, value):
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path, index_col = 0)
        if dataset_name not in df.index:
            df.loc[dataset_name] = pd.NA
        if number_name not in df.columns:
            df[number_name] = pd.NA
        df.loc[dataset_name, number_name] = value
    else:
        df = pd.DataFrame(index = [dataset_name], columns = [number_name])
        df.loc[dataset_name, number_name] = value
    return df.to_csv(csv_path, index = True)

def invalid_mol_filter(base_path, dataset_name):
    dataset = pd.read_csv(f'{base_path}/molnet_dataset/{dataset_name}.csv')
    dataset_smiles = dataset['smiles']
    n_beginning = len(dataset)

    n_invalid = 0
    index_invalid = []
    invalid_mol = []
    invalid_smiles = []

    for i in range(n_beginning):
        mol = Chem.MolFromSmiles(dataset_smiles[i])
        if mol == None:
            invalid_mol.append(mol)
            invalid_smiles.append(dataset_smiles[i])
            n_invalid += 1
            index_invalid.append(i)

    filtered_dataset = dataset.drop(index_invalid, axis=0)
    csv_path = f'{base_path}/data_curation_summary.csv'
    num_molecule_to_csv(csv_path, dataset_name, 'n_beginning', n_beginning)
    num_molecule_to_csv(csv_path, dataset_name, 'n_invalid', n_invalid)

    os.makedirs(f'{base_path}/filtered_invalid/', exist_ok=True)
    filtered_dataset.to_csv(f'{base_path}/filtered_invalid/{dataset_name}.csv', index=False)

    os.makedirs(f'{base_path}/{dataset_name}/', exist_ok=True)

    index_file = f'{base_path}/{dataset_name}/{dataset_name}_invalid_indices.csv'
    pd.DataFrame({'invalid_index': index_invalid}).to_csv(index_file, index=False)

    mol_file = f'{base_path}/{dataset_name}/{dataset_name}_invalid_mol.csv'
    pd.DataFrame({'invalid_mol': invalid_mol}).to_csv(mol_file, index=False)

    smiles_file = f'{base_path}/{dataset_name}/{dataset_name}_invalid_smiles.csv'
    pd.DataFrame({'invalid_smiles': invalid_smiles}).to_csv(smiles_file, index=False)

    return invalid_mol, index_invalid, index_invalid

def drop_replicate(base_path, dataset_name, task):
    dataset = pd.read_csv(f'{base_path}/filtered_invalid/{dataset_name}.csv')
    dataset['canonical_smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    replicate_index = dataset.index[dataset['canonical_smiles'].duplicated(keep=False)]
    n_replicate = len(replicate_index)
    label = dataset.columns.drop(['canonical_smiles', 'smiles']).to_list()

    os.makedirs(f'{base_path}/{dataset_name}/', exist_ok=True)

    to_drop = []
    for idx1 in replicate_index:
        if idx1 in to_drop:
            continue
        ref_smiles = dataset.loc[idx1, 'canonical_smiles']
        ref_labels = dataset.loc[idx1,label]
        for idx2 in replicate_index:
            if idx2 in to_drop or idx1 == idx2:
             continue
            rep_smiles = dataset.loc[idx2, 'canonical_smiles']
            rep_labels = dataset.loc[idx2, label]
            if ref_smiles == rep_smiles:
                if task == 'regression':
                    to_drop.append(idx1)
                    to_drop.append(idx2)
                elif task == 'classification':
                    if ref_labels.equals(rep_labels):
                        to_drop.append(idx2)
                    else:
                        to_drop.append(idx1)
                        to_drop.append(idx2)
    n_filtered_replicate = len(to_drop)
    drop_replicate_dataset = dataset.drop(to_drop, axis=0).drop(['canonical_smiles'], axis=1)
    n_final = len(drop_replicate_dataset)
    
    replicate_indices_file = f'{base_path}/{dataset_name}/{dataset_name}_replicate_indices.csv'
    pd.DataFrame({'replicate_index': replicate_index}).to_csv(replicate_indices_file, index=False)

    to_drop_indices_file = f'{base_path}/{dataset_name}/{dataset_name}_to_drop_indices.csv'
    pd.DataFrame({'to_drop_index': to_drop}).to_csv(to_drop_indices_file, index=False)

    csv_path = f'{base_path}/data_curation_summary.csv'
    os.makedirs(f'{base_path}/curated_dataset/', exist_ok=True)

    num_molecule_to_csv(csv_path,dataset_name,'n_replicate',n_replicate)
    num_molecule_to_csv(csv_path, dataset_name, 'n_filtered_replicate', n_filtered_replicate)
    num_molecule_to_csv(csv_path, dataset_name, 'n_final', n_final)
    drop_replicate_dataset.to_csv(f'{base_path}/curated_dataset/{dataset_name}.csv', index=False)

    return drop_replicate_dataset

def class_balance(base_path, dataset_name,task):
    if task == 'regression':
        assert False
    dataset = pd.read_csv(f'{base_path}/curated_dataset/{dataset_name}.csv')
    labels = dataset.drop('smiles',axis =1).columns
    num_tasks = len(labels)

    percent_positive = []
    for label in labels:
        counts = dataset[label].value_counts()
        positive = counts.get(1,0)
        negative = counts.get(0,0)
        total = positive + negative

        score = positive/total*100
        percent_positive.append(score)

    assert num_tasks == len(percent_positive)
    avg_percent_positive = round(np.mean(percent_positive),2)

    csv_path = f'{base_path}/data_curation_summary.csv'
    os.makedirs(f'{base_path}/curated_dataset/', exist_ok=True)

    num_molecule_to_csv(csv_path,dataset_name,'percent_positive',avg_percent_positive)
    return avg_percent_positive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run Data Curation')
    parser.add_argument('--dataset_name', type =str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()
    
    invalid_mol_filter(args.base_path, args.dataset_name)
    drop_replicate(args.base_path, args.dataset_name, args.task)
    class_balance(args.base_path, args.dataset_name, args.task)