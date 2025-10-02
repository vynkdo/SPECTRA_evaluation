import deepchem as dc
import numpy as np
import os
import umap.umap_ as umap
import pickle
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans

def convert_to_numpy_dataset(dataset_name, base_path):
    df = pd.read_csv(f"{base_path}/dataset/curated_dataset/{dataset_name}.csv")

    mfp = []
    for i in range(len(df['smiles'])):
        mol = Chem.MolFromSmiles(df['smiles'][i])
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        mfp.append(fp)

    numpy_dataset = dc.data.NumpyDataset(y=df.drop(columns='smiles').values, ids=df['smiles'].values, X = mfp)
    return numpy_dataset, df

def generate_random_and_scaffold_splits(dataset_name, base_path):
    splitter_dic = {"random": dc.splits.RandomSplitter,
                    "scaffold": dc.splits.ScaffoldSplitter}
    dataset, df = convert_to_numpy_dataset(dataset_name, base_path)

    for split_type, splitter_type in splitter_dic.items():
        save_dir = os.path.join(base_path, split_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        splitter = splitter_type()

        for index, value in enumerate([42, 43, 44, 45, 46]):
            train, test = splitter.train_test_split(dataset, frac_train = 0.8, seed = value)
            train_smiles = set(train.ids)
            test_smiles = set(test.ids)

            train_indices = [index for index, smiles in enumerate(dataset.ids) if smiles in train_smiles]
            test_indices = [index for index, smiles in enumerate(dataset.ids) if smiles in test_smiles]

            print(f"train: {train_indices}")
            print(f"test: {test_indices}")
            print(len(set(train_indices) & set(test_indices)))

            assert len(set(train_indices) & set(test_indices)) == 0
            assert len(set(train_indices + test_indices)) == len(dataset.ids)

            print(f"{dataset_name} split {split_type} {index}")

            with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_train_split_{index}.pkl"), "wb") as f:
                pickle.dump(train_indices, f)

            with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_test_split_{index}.pkl"), "wb") as f:
                pickle.dump(test_indices, f)
    return print("Random and scaffold splits done.")

def generate_umap_splits(dataset_name, base_path, n_clusters = 7):
    dataset, df = convert_to_numpy_dataset(dataset_name, base_path)
    mfp_dataset = dataset.X
    test_size = round(0.2 * len(mfp_dataset), 0)

    umap_save_dir = os.path.join(base_path, f"umap/{dataset_name}")
    os.makedirs(umap_save_dir, exist_ok=True)

    for index, value in enumerate([42, 43, 44, 45, 46]):
        mfp_umap = umap.UMAP(n_neighbors = 15, n_components = 2, transform_seed = value).fit_transform(mfp_dataset)
        kmeans = KMeans(n_clusters = n_clusters, random_state = value)
        cluster_labels = kmeans.fit_predict(mfp_umap)

        cluster_index, counts = np.unique(cluster_labels, return_counts=True)
        difference_list = [abs(test_size - i) for i in counts]
        min_cluster_index = difference_list.index(min(difference_list))
        umap_test_indices = np.where(cluster_labels == min_cluster_index)[0]
        umap_train_indices = np.where(cluster_labels != min_cluster_index)[0]

        assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
        assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfp_dataset)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_train_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_train_indices, f)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_test_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_test_indices, f)
    return print("UMAP splits done.")

if __name__ == "__main__":
    datasets = ['bbbp','clintox','delaney','sider']
    base_path = "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"

    for dataset in datasets:
        generate_umap_splits(dataset, base_path)