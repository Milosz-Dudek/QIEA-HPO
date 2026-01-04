"""Search space for all the models."""
from typing import Dict, Any


def get_search_space(model_type: str) -> Dict[str, Any]:
    """
    Return a unified search space for each model.
    Types: 'xgboost', 'svm_rbf', 'mlp', 'cnn_cifar10'
    """
    if model_type == "xgboost":
        return {
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "n_estimators": {"type": "int", "low": 50, "high": 400},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        }
    elif model_type == "svm_rbf":
        return {
            "C": {"type": "float", "low": 1e-2, "high": 1e3, "log": True},
            "gamma": {"type": "float", "low": 1e-4, "high": 1e0, "log": True},
        }
    elif model_type == "mlp":
        return {
            "hidden_dim": {"type": "int", "low": 32, "high": 256},
            "num_layers": {"type": "int", "low": 1, "high": 3},
            "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "batch_size": {"type": "int", "low": 32, "high": 128},
        }
    elif model_type == "cnn_cifar10":
        return {
            "lr": {"type": "float", "low": 3e-4, "high": 3e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
            "weight_decay": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        }
    elif model_type == "cnn_mnist":
        return {
            # Architecture
            "num_conv_blocks": {"type": "int", "low": 1, "high": 3},
            "base_channels": {"type": "categorical", "choices": [16, 32, 64]},
            "fc_dim": {"type": "categorical", "choices": [64, 128, 256]},
            "use_batchnorm": {"type": "categorical", "choices": [True, False]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},

            # Training details
            "optimizer_type": {"type": "categorical", "choices": ["sgd", "adam"]},
            "lr": {"type": "float", "low": 1e-4, "high": 5e-2, "log": True},
            "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},

            # Resource dimension for multi-objective acc vs epochs
            "epochs": {"type": "int", "low": 3, "high": 10},
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
