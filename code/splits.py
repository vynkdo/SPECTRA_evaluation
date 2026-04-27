import deepchem as dc
import numpy as np
import os
import umap.umap_ as umap
import pickle
import random
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from sklearn.cluster import KMeans
import argparse
import matplotlib.pyplot as plt

def convert_to_numpy_dataset(dataset_name, base_path):
    df = pd.read_csv(f"{base_path}/dataset/curated_dataset/{dataset_name}.csv")

    mfp = []
    for i in range(len(df['smiles'])):
        mol = Chem.MolFromSmiles(df['smiles'][i])
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        mfp.append(fp)

    numpy_dataset = dc.data.NumpyDataset(y=df.drop(columns='smiles').values, ids=df['smiles'].values, X = mfp)
    return numpy_dataset, df, mfp

def generate_random_splits(dataset_name, base_path, split_type="random"):
    dataset, df, mfp = convert_to_numpy_dataset(dataset_name, base_path)

    save_dir = os.path.join(base_path, split_type, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    splitter = dc.splits.RandomSplitter()

    for index, value in enumerate([42, 43, 44, 45, 46]):
        train, test = splitter.train_test_split(dataset, frac_train=0.8, seed=value)
        train_smiles = set(train.ids)
        test_smiles = set(test.ids)

        train_indices = [index for index, smiles in enumerate(dataset.ids) if smiles in train_smiles]
        test_indices = [index for index, smiles in enumerate(dataset.ids) if smiles in test_smiles]

        assert len(set(train_indices) & set(test_indices)) == 0
        assert len(set(train_indices + test_indices)) == len(dataset.ids)

        print(f"{dataset_name} split random {index}")

        with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_train_split_{index}.pkl"), "wb") as f:
            pickle.dump(train_indices, f)

        with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_test_split_{index}.pkl"), "wb") as f:
            pickle.dump(test_indices, f)


def generate_scaffold_splits(dataset_name, base_path, split_type="scaffold"):
    dataset, df, mfp = convert_to_numpy_dataset(dataset_name, base_path)

    # calculate scaffolds from Molecules (deterministic)
    scaffolds = []
    for smi in df['smiles']:
        mol = Chem.MolFromSmiles(smi)
        scaffolds.append(Chem.MolToSmiles(GetScaffoldForMol(mol)))

    scaffold_to_ix_map = {
        scaf: [ix for ix, s in enumerate(scaffolds) if s == scaf] for scaf in scaffolds
    }

    save_dir = os.path.join(base_path, split_type, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for index, value in enumerate([42, 43, 44, 45, 46]):
        random.seed(value)

        train_prop = 0.8
        tol = 0.01 # tolerance on proportion away from desired train proportion

        train_indices = []
        test_indices = []

        while abs((len(train_indices) / len(scaffolds)) - train_prop) > tol:
            train_indices = []
            test_indices = []
            shuffled_unique_scaffolds = random.sample(list(scaffold_to_ix_map.keys()), k=len(scaffold_to_ix_map))

            print((train_prop*df.shape[0]))
            while len(train_indices) < (train_prop*df.shape[0]):
                cur_scaf = shuffled_unique_scaffolds.pop(0)
                train_indices += scaffold_to_ix_map[cur_scaf]

            for remaining_scaffolds in shuffled_unique_scaffolds:
                test_indices += scaffold_to_ix_map[remaining_scaffolds]

            print(f"Train proportion: {len(train_indices) / len(scaffolds)}")
            print(f"Test proportion: {len(test_indices) / len(scaffolds)}")


        print(f"{dataset_name} split {split_type} {index}")

        assert len(set(train_indices) & set(test_indices)) == 0
        assert len(set(train_indices + test_indices)) == len(dataset.ids)

        print(test_indices)

        with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_train_split_{index}.pkl"), "wb") as f:
            pickle.dump(train_indices, f)

        with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_test_split_{index}.pkl"), "wb") as f:
            pickle.dump(test_indices, f)

def generate_umap_splits(dataset_name, base_path, n_clusters = 7):
    dataset, df, mfp = convert_to_numpy_dataset(dataset_name, base_path)
    mfp_dataset = dataset.X
    test_size = round(0.2 * len(mfp_dataset), 0)

    umap_save_dir = os.path.join(base_path, f"splits/umap/{dataset_name}")
    umap_save_dir_plot = os.path.join(base_path, f"splits/umap/{dataset_name}/plot")
    os.makedirs(umap_save_dir, exist_ok=True)
    os.makedirs(umap_save_dir_plot, exist_ok=True)

    for index, value in enumerate([42, 43, 44, 45, 46]):
        mfp_umap = umap.UMAP(n_neighbors = 15, n_components = 2, transform_seed = value).fit_transform(mfp_dataset)
        kmeans = KMeans(n_clusters = n_clusters, random_state = value)
        cluster_labels = kmeans.fit_predict(mfp_umap)

        plt.figure(figsize=(6,6))
        plt.scatter(mfp_umap[:, 0], mfp_umap[:, 1], c=cluster_labels,s=50,alpha=1.0)
        plt.title(f"{dataset_name} - UMAP embedding {index}")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.savefig(os.path.join(umap_save_dir_plot, f"{dataset_name}_umap_{index}.png"), dpi=400)
        plt.close()

        cluster_index, counts = np.unique(cluster_labels, return_counts=True)
        difference_list = [abs(test_size - i) for i in counts]
        min_cluster_index = difference_list.index(min(difference_list))
        umap_test_indices = (np.where(cluster_labels == min_cluster_index)[0]).tolist()
        umap_train_indices = (np.where(cluster_labels != min_cluster_index)[0]).tolist()
    
        assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
        assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfp_dataset)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_train_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_train_indices, f)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_test_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_test_indices, f)
    print(f"UMAP splits {dataset_name} done.")

if __name__ == "__main__":
    base_path = '../' # the main github directory

    for dname in ['bace','bbbp','clintox','delaney','freesolv','hiv','lipo','sider','tox21']:
        pass
        # Uncomment to re-run all splits; takes time
        # generate_random_splits(dname, base_path)
        # generate_scaffold_splits(dname, base_path)
        # generate_umap_splits(dname, base_path)
