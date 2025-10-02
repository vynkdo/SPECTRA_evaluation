import pandas as pd
from rdkit import Chem
from pathlib import Path
import numpy as np
import os

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
    dataset = pd.read_csv(f'{base_path}/clean_dataset/{dataset_name}.csv')
    dataset_smiles = dataset['smiles']
    n_beginning = len(dataset)

    n_invalid = 0
    index_invalid = []

    for i in range(n_beginning):
        mol = Chem.MolFromSmiles(dataset_smiles[i])
        if mol == None:
            n_invalid += 1
            index_invalid.append(i)

    filtered_dataset = dataset.drop(index_invalid, axis=0)
    csv_path = f'{base_path}/data_curation_summary.csv'
    num_molecule_to_csv(csv_path, dataset_name, 'n_beginning', n_beginning)
    num_molecule_to_csv(csv_path, dataset_name, 'n_invalid', n_invalid)

    os.makedirs(f'{base_path}/filtered_invalid/', exist_ok=True)
    filtered_dataset.to_csv(f'{base_path}/filtered_invalid/{dataset_name}.csv', index=False)
    return filtered_dataset

def drop_duplicate(base_path, dataset_name, task):
    dataset = pd.read_csv(f'{base_path}/filtered_invalid/{dataset_name}.csv')
    dataset['canonical_smiles'] = dataset['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    replicate_index = dataset.index[dataset['canonical_smiles'].duplicated(keep=False)]
    n_replicate = len(replicate_index)
    label = dataset.columns.drop(['canonical_smiles', 'smiles']).to_list()

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
    n_filtered_duplicate = len(to_drop)
    drop_duplicate_dataset = dataset.drop(to_drop, axis=0).drop(['canonical_smiles'], axis=1)
    n_final = len(drop_duplicate_dataset)

    csv_path = f'{base_path}/data_curation_summary.csv'
    os.makedirs(f'{base_path}/curated_dataset/', exist_ok=True)

    num_molecule_to_csv(csv_path,dataset_name,'n_replicate',n_replicate)
    num_molecule_to_csv(csv_path, dataset_name, 'n_filtered_duplicate', n_filtered_duplicate)
    num_molecule_to_csv(csv_path, dataset_name, 'n_final', n_final)
    drop_duplicate_dataset.to_csv(f'{base_path}/curated_dataset/{dataset_name}.csv', index=False)

    return drop_duplicate_dataset

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
    base_path = '/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies/dataset'
    classification_datasets = ['bace','bbbp','clintox','hiv','sider','tox21']
    regression_datasets = ['delaney','freesolv','lipo']
    for dataset_name in classification_datasets:
        class_balance(base_path,dataset_name,'classification')