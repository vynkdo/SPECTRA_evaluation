import os
import deepchem as dc
from spectrae import Spectra, SpectraDataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tqdm import tqdm
from os.path import join
import pickle
from pathlib import Path
import argparse


class MolnetDataset(SpectraDataset):
  def parse(self, dataset):
    return dataset

  def __len__(self):
    return len(self.samples)

  def sample_to_index(self,sample):
    if not hasattr(self, 'index_to_sequence'):
      print('Generating index to sequence')
      self.index_to_sequence = {}
      for i in tqdm(range(len(self.samples))):
        x = self.__getitem__(i)
        self.index_to_sequence[x] = i
    return self.index_to_sequence[sample]

  def __getitem__(self, idx):
    return self.samples[idx]

class MolnetTanimotoSpectra(Spectra):
  def spectra_properties(self, sample_one, sample_two):
    return TanimotoSimilarity(sample_one, sample_two)

  def cross_split_overlap(self, train, test):
    average_similarity = []
    for i in train:
      for j in test:
        average_similarity.append(self.spectra_properties(i,j))
    return np.mean(average_similarity)

class MolnetHammingSpectra(Spectra):
  def spectra_properties(self, sample_one, sample_two):
      return np.sum(sample_one == sample_two)/1024

  def cross_split_overlap(self, train, test):
    average_similarity = []
    for i in train:
      for j in test:
        average_similarity.append(self.spectra_properties(i,j))
    return np.mean(average_similarity)

def convert_to_morgan_fingerprint(dataset_name, base_path):
  dataset = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
  dataset_smiles = dataset['smiles']

  mfp = []
  for i in range(len(dataset_smiles)):
    mol = Chem.MolFromSmiles(dataset_smiles[i])
    print(i)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    mfp.append(fp)

  return mfp

def generate_spectra_tanimoto_splits(dataset_name, spectra_parameters, base_path):
  mfp = convert_to_morgan_fingerprint(dataset_name, base_path)
  save_dir = f'{base_path}/splits/spectra_tanimoto/{dataset_name}'
  os.makedirs(save_dir, exist_ok=True)

  spectra_dataset = MolnetDataset(mfp, f'{dataset_name}')
  tanimoto_spectra = MolnetTanimotoSpectra(spectra_dataset, binary=False)
  tanimoto_spectra.pre_calculate_spectra_properties(f'{dataset_name}', force_recalculate=False)
  tanimoto_spectra.generate_spectra_splits(**spectra_parameters)

  stats = tanimoto_spectra.return_all_split_stats()
  stats_df = pd.DataFrame(stats).sort_values(by='SPECTRA_parameter', ascending=True)

  return stats_df

def random_scaffold_umap_cross_split_overlap(dataset_name, split_type, base_path):
  mfp = convert_to_morgan_fingerprint(dataset_name, base_path)
  print(len(mfp))
  for i in range(5):
    with open(join(base_path,
                 f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_train_split_{i}.pkl'),
            'rb') as f:
      train_indices = pickle.load(f)
      print(train_indices)

    with open(join(base_path,
                 f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_test_split_{i}.pkl'),
            'rb') as f:
      test_indices = pickle.load(f)
      print(test_indices)

    train = [int(mfp[m]) for m in train_indices]
    print(len(train))
    test = [int(mfp[n]) for n in test_indices]
    print(len(test))

    average_similarity = []
    for train_mfp in train:
      for test_mfp in test:
        average_similarity.append(TanimotoSimilarity(train_mfp, test_mfp))

    cross_split_overlap = np.mean(average_similarity)

    dir = f'splits/{split_type}/{dataset_name}'
    filename = f'{dataset_name}_{split_type}_cross_split_overlap.csv'
    csv_path = join(base_path, dir, filename)

    row = pd.DataFrame({'train_size': len(train),
                         'test_size': len(test),
                        'cross_split_overlap': cross_split_overlap}, index=[f'{dataset_name}_{split_type}_{i}'])
    if Path(csv_path).exists():
      df = pd.read_csv(csv_path, index_col=0)
      col_order = df.columns.tolist()
      row = row[col_order]
      df = pd.concat([df, row], axis=0)
    else:
      df = row

    df.to_csv(csv_path)
  return print(f'Done with {dataset_name}_{split_type}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run SPECTRA Taniomoto Splits')
    parser.add_argument('--dataset_name', type =str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--split_type', type=str, required=True )
    args = parser.parse_args()
    spectra_parameters = {'number_repeats': 3,
                          'random_seed': [42, 44, 46],
                          'spectral_parameters': ["{:.2f}".format(i) for i in np.arange(0, 1.05, 0.05)],
                          'force_reconstruct':False,
                          }
    
    random_scaffold_umap_cross_split_overlap(args.dataset_name, args.split_type, args.base_path)
 