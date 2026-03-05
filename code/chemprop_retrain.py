import deepchem as dc
from chemprop import featurizers, data, nn, models
import pandas as pd
import numpy as np
import pickle
from lightning import pytorch as pl
from os.path import join
from pathlib import Path
import torch
import splits
import os
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import CheckpointConfig, RunConfig,ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
import tomllib

def train(molecule_datapoint, task_type, num_tasks, train_indices, test_indices):
    # Define Chemprop Model
    graph_transform = nn.GraphTransform(
        V_transform=torch.nn.Identity(),
        E_transform=torch.nn.Identity()
    )
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(graph_transform=graph_transform)
    agg = nn.NormAggregation()
    batch_norm = False

    if task_type == 'classification':
        ffn = nn.BinaryClassificationFFN(n_tasks=num_tasks)
        metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score(), nn.metrics.BinaryAccuracy(), nn.metrics.BinaryAUPRC()]
    elif task_type == 'regression':
        ffn = nn.RegressionFFN()
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    else:
        raise ValueError(f"task_type must be either 'classification' or 'regression'")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=50)

    train_dset = data.MoleculeDataset([molecule_datapoint[j] for j in train_indices], featurizer=featurizer)
    test_dset = data.MoleculeDataset([molecule_datapoint[j] for j in test_indices], featurizer=featurizer)

    y_train = np.array([train_dset[j].y for j in range(len(train_dset))])
    y_test = np.array([test_dset[j].y for j in range(len(test_dset))])

    if ffn == nn.BinaryClassificationFFN():
        assert np.any(y_train == 0) and np.any(y_train == 1)
        assert np.any(y_test == 0) and np.any(y_test == 1)

    train_loader = data.build_dataloader(train_dset)
    test_loader = data.build_dataloader(test_dset)

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    trainer.fit(mpnn, train_loader)
    results = trainer.test(mpnn, test_loader)

    return results

def metrics_to_csv(csv_path, results, row_name):
    row = pd.DataFrame(results, index= [row_name])
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path, index_col=0)
        col_order = df.columns.tolist()
        row = row[col_order]
        df = pd.concat([df, row], axis=0)
    else:
        df = row
    df.to_csv(csv_path)

def chemprop(dataset_name, split_type, base_path, task_type, config):
    # Load Dataset
    df = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
    smis = df.loc[:, 'smiles'].values
    ys = df.drop(columns='smiles').values
    num_tasks = len(df.columns.drop('smiles'))
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    # Save Directory
    dir = f'metrics/chemprop/{split_type}'
    output = Path(base_path) / dir
    output.mkdir(parents=True, exist_ok=True)
    filename = f'{dataset_name}_{split_type}_metrics.csv'
    csv_path = join(base_path, dir, filename)

    if split_type in ['spectra_tanimoto','spectra_hamming','inverse_spectra_hamming']:
        print(f'Starting with {dataset_name}_{split_type}')
        root = Path(base_path) / "splits" / split_type / f"{dataset_name}_SPECTRA_splits"
        for parameter in range(21):
            parameter = f'{parameter/20:.2f}'
            for i in range(1):
                sp_dir = root / f"SP_{parameter}_{i}"
                train_file = sp_dir / "train.pkl"
                test_file = sp_dir / "test.pkl"
                stats_file = sp_dir / "stats.pkl"

                if not (train_file.exists() and test_file.exists() and stats_file.exists()):
                    continue
                with train_file.open("rb") as f:
                    train_indices = pickle.load(f)
                with test_file.open("rb") as f:
                    test_indices = pickle.load(f)
                with stats_file.open("rb") as f:
                    stat_info = pickle.load(f)

                results = train(all_data, task_type, num_tasks, train_indices, test_indices,config)
                merge = {**stat_info, **results[0]}
                metrics_to_csv(csv_path, merge, f'{dataset_name}_{split_type}_{parameter}_{i}')
                print(f'Done with {dataset_name}_{split_type}_{parameter}_{i}')
        print(f'Complete {dataset_name}_{split_type}')
    else:
        print(f'Starting with {dataset_name}_{split_type}')
        for i in range(1):
            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_train_split_{i}.pkl'),
                      'rb') as f:
                train_indices = pickle.load(f)

            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_test_split_{i}.pkl'),
                      'rb') as f:
                test_indices = pickle.load(f)

            results = train(all_data, task_type, num_tasks, train_indices, test_indices,config)

            metrics_to_csv(csv_path, results, f'{dataset_name}_{split_type}_{i}')
            print(f'Done with {dataset_name}_{split_type}_{i}')
        print(f'Complete with {dataset_name}_{split_type}')
    return None

if __name__ == '__main__':
    classification_datasets = ['tox21'] 
    regression_datasets = []
    split_types = ['spectra_tanimoto']
    base_path = "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"
    # with open(f'{base_path}/chemprop_hpopt/clintox/best_config.toml', "rb") as f:  # note: open in binary mode
    #     config = tomllib.load(f)
    for dataset in classification_datasets:
        for split_type in split_types:
             chemprop(dataset,split_type,base_path,'classification')