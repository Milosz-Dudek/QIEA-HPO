"""
Tabular benchmark tasks (UCI-like) for HPO experiments.
"""

from typing import Any
import time

from loguru import logger

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def get_tabular_dataset(name: str):
    """
    Load and standardize a small tabular dataset.

    Parameters
    ----------
    name : {"breast_cancer", "wine"}
        Name of the dataset.

    Returns
    -------
    X_train, X_val, y_train, y_val : arrays
        Train/validation splits.
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError(f"Unknown dataset {name}")
    X = data.data
    y = data.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val


def make_xgboost_objective(
        dataset_name: str,
) -> tuple[callable, dict[str, Any]]:
    """
    Build an objective function for XGBoost on a given tabular dataset.
    """
    X_train, X_val, y_train, y_val = get_tabular_dataset(dataset_name)
    logger.info(f"Prepared XGBoost task on dataset={dataset_name}")

    def objective(params: dict[str, Any]):
        """Main objective"""
        start = time.time()
        clf = XGBClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        elapsed = time.time() - start
        # NOTE: expose both acc and time
        return acc, {"time": elapsed, "acc": acc}

    return objective, {"dataset": dataset_name, "model": "xgboost"}


def make_svm_objective(
        dataset_name: str,
) -> tuple[callable, dict[str, Any]]:
    """
    Build an objective function for RBF SVM on a given tabular dataset.
    """
    X_train, X_val, y_train, y_val = get_tabular_dataset(dataset_name)
    logger.info(f"Prepared SVM task on dataset={dataset_name}")

    def objective(params: dict[str, Any]):
        """Main objective"""
        start = time.time()
        clf = SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel="rbf",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        elapsed = time.time() - start
        return acc, {"time": elapsed, "acc": acc}

    return objective, {"dataset": dataset_name, "model": "svm_rbf"}


def make_mlp_objective(
        dataset_name: str,
) -> tuple[callable, dict[str, Any]]:
    """
    Build an objective function for a shallow MLP on a given tabular dataset.
    """
    X_train, X_val, y_train, y_val = get_tabular_dataset(dataset_name)
    logger.info(f"Prepared MLP task on dataset={dataset_name}")

    def objective(params: dict[str, Any]):
        """Main objective"""
        start = time.time()
        hidden = params["hidden_dim"]
        layers = params["num_layers"]
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden,) * layers,
            learning_rate_init=params["lr"],
            batch_size=params["batch_size"],
            max_iter=100,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        elapsed = time.time() - start
        info = {
            "time": elapsed,
            "acc": acc,  # optional but handy
            "epochs": 1,  # single pass / fit
        }
        return acc, info

    return objective, {"dataset": dataset_name, "model": "mlp"}
