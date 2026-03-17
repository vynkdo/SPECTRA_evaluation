import os
import json
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger # tragic that this is the best way

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Lasso # Linear model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error


SPLIT_DIR = "../splits_data/chemprop_data/"
DATASET_DIR = "../dataset/curated_dataset"

# Suppress warning messages, because they're unfixable
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def load_dataset(dataset):
    dataset_path = os.path.join(DATASET_DIR, dataset + ".csv")
    df = pd.read_csv(dataset_path)

    X_orig = df["smiles"]
    y_orig = df.drop(columns=["smiles"])

    X = np.zeros((len(X_orig), 1024))
    y = np.zeros((len(y_orig), len(y_orig.columns)))

    for i, smi in enumerate(X_orig):
        fp = rdFingerprintGenerator.GetMorganGenerator(
                        radius=2, fpSize=1024).GetFingerprint(
                        Chem.MolFromSmiles(smi))
        X[i] = fp

    for i in range(len(y_orig)):
        for j in range(len(y_orig.columns)):
            y[i, j] = y_orig.iloc[i, j]


    return X, y


def load_split_indices_non_spectra(dataset, split_type, split_num):
    # get the indices of train vs test
    fn = f"data_{split_num}.json"
    data_path = os.path.join(SPLIT_DIR, split_type, dataset, fn)

    with open(data_path, "rb") as f:
        indices = json.load(f)

    train_indices = indices[0]["train"]
    test_indices = indices[0]["test"]

    return train_indices, test_indices


def load_split(dataset, split_type, split_num):
    X, y = load_dataset(dataset)
    train_indices, test_indices = load_split_indices_non_spectra(dataset, split_type, split_num)

    return X[train_indices,:], X[test_indices,:], y[train_indices,:], y[test_indices,:]


def build_models():
    # Kernel choices and other factors are
    # as described in PMC5868307,
    # the MoleculeNet paper,
    # except for the linear regression
    # which for some reason was not included
    # in their paper at all.

    logreg_lasso = LogisticRegression(
        penalty="l1", # LASSO
        solver="saga"
    )
    linreg_lasso = Lasso()

    random_forest = RandomForestClassifier()
    xgb = GradientBoostingClassifier()

    svm = SVC(kernel="rbf")
    krr = KernelRidge(kernel="rbf")

    return {
        "LogReg": logreg_lasso,
        "LinReg": linreg_lasso,
        "RF": random_forest,
        "XGB": xgb,
        "SVM": svm,
        "KRR": krr
    }


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds)
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probs)

    return metrics


def run_pipeline():
    dataset_model_dict = {
        #"bace": ["LogReg", "RF", "XGB", "SVM"], # binary classification
        #"bbbp": ["LogReg", "RF", "XGB", "SVM"], # binary classification
        "clintox": ["RF"], # multitask binary classification
        # TODO fix metric reporting for regression tasks
        # TODO add F1, ROC AUC metrics instead of just accuracy
        #"delaney": ["LinReg", "RF", "XGB", "KRR"], # regression
        #"freesolv": ["LinReg", "RF", "XGB", "KRR"], # regression
        #"lipo": ["LinReg", "RF", "XGB", "KRR"], # regression
        # TODO fix metric reporting for multitask tasks
        # TODO see if we can get logreg (no longer implemented by deepchem)
        # TODO or XGB/variant or SVM/variant working too
        #"sider": ["RF"], # multitask binary classification
        #"tox21": ["RF"] # multitask binary classification
    }

    split_types = ["random", "scaffold", "umap"]
    models = build_models()
    results = []

    for dataset in dataset_model_dict.keys():
        for split_type in split_types:
            for split_num in range(5):
                X_train, X_test, y_train, y_test = load_split(dataset, split_type, split_num)

                if y_train.shape[1] == 1 and y_test.shape[1] == 1:
                    y_train = y_train.ravel()
                    y_test = y_test.ravel()

                print(f"Running split: {dataset} {split_type} {split_num}")

                for model_name in dataset_model_dict[dataset]:
                    print(f"Running for model {model_name}")
                    model = models[model_name]
                    model.fit(X_train, y_train)
                    # TODO update metrics
                    metrics = {} #evaluate_model(model, X_test, y_test)

                    result = {
                        "dataset": dataset,
                        "split_type": split_type,
                        "split_num": split_num,
                        "model": model_name,
                        **metrics
                    }

                    results.append(result)

                print(results)

    results_df = pd.DataFrame(results)

    print("\nResults per split:")
    print(results_df)

    print("\nAverage performance:")
    print(results_df.groupby("model").mean(numeric_only=True))

    return results_df


if __name__ == "__main__":
    # For all datasets, I checked distribution of y values
    # especially for regression, to see if normalization was needed
    # It seemed that log normalization was already applied where necessary
    run_pipeline()
