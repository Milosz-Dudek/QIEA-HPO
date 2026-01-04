"""
Entry point for running a single HPO experiment.

This script wires together:
- A chosen optimizer (QIEA, random, GA, TPE, ASHA),
- A chosen task (tabular XGBoost/SVM/MLP or CIFAR-10 CNN),
- Single- or multi-objective mode for QIEA.

Results are written as JSON (trials + best) and logs are emitted via loguru.
"""

import argparse
import os
from typing import Any

from loguru import logger

from modules.optimizers.qiea import QIEAOptimizer
from modules.optimizers.random_search import RandomSearchOptimizer
from modules.optimizers.ga import GAOptimizer
from modules.optimizers.optuna_tpe import OptunaTPEOptimizer
from modules.optimizers.optuna_asha import OptunaASHAOptimizer

from modules.utils.search_space import get_search_space
from modules.utils.logging_utils import save_trials, save_best, save_pareto_front, setup_logger
from modules.utils.seed import set_global_seed

from modules.tasks.tabular_tasks import (
    make_xgboost_objective,
    make_svm_objective,
    make_mlp_objective,
)
from modules.tasks.vision_tasks import make_cifar10_objective, make_mnist_objective


def build_objective(task: str):
    """
    Construct an objective function and metadata for a given task name.

    Parameters
    ----------
    task : str
        One of:
        - "xgboost_breast_cancer"
        - "xgboost_wine"
        - "svm_breast_cancer"
        - "mlp_wine"
        - "cnn_cifar10"

    Returns
    -------
    objective : callable
        Function mapping hyperparameter dict -> (score, info_dict).
        By convention, score is validation accuracy in [0, 1],
        and info_dict at least contains {"time": <seconds>}.
    meta : dict
        Small dictionary with dataset/model descriptors (for logging/analysis).
    """
    if task == "xgboost_breast_cancer":
        return make_xgboost_objective("breast_cancer")
    elif task == "xgboost_wine":
        return make_xgboost_objective("wine")
    elif task == "svm_breast_cancer":
        return make_svm_objective("breast_cancer")
    elif task == "mlp_wine":
        return make_mlp_objective("wine")
    elif task == "cnn_cifar10":
        return make_cifar10_objective()
    elif task == "cnn_mnist":
        return make_mnist_objective()
    else:
        raise ValueError(f"Unknown task {task}")


def build_search_space(task: str) -> dict[str, Any]:
    """
    Obtain the search space for a given task.

    Parameters
    ----------
    task : str
        Task name used in build_objective.

    Returns
    -------
    dict
        Search space description consumed by the optimizers.
    """
    if task.startswith("xgboost"):
        return get_search_space("xgboost")
    elif task.startswith("svm"):
        return get_search_space("svm_rbf")
    elif task.startswith("mlp"):
        return get_search_space("mlp")
    elif task.startswith("cnn_cifar10"):
        return get_search_space("cnn_cifar10")
    elif task.startswith("cnn_mnist"):
        return get_search_space("cnn_mnist")
    else:
        raise ValueError(f"Unknown task {task}")


def build_optimizer(
        name: str,
        search_space: dict[str, Any],
        n_trials: int,
        seed: int,
        objective_type: str,
        mo_objectives: tuple[str, str] | None,
):
    """
    Instantiate the requested optimizer.

    Parameters
    ----------
    name : {"qiea", "random", "ga", "tpe", "asha"}
        Optimizer identifier.
    search_space : dict
        Search space specification (categorical/int/float).
    n_trials : int
        Evaluation budget (maximum number of objective calls).
    seed : int
        Random seed passed into the optimizer (for reproducibility).
    objective_type : {"single", "multi"}
        Objective mode. Currently, only QIEA consumes "multi" explicitly
        (via its `multi_objective` flag); baselines are single-objective.
    mo_objectives : {}
        Objectives for Pareto to be used.

    Returns
    -------
    optimizer : BaseOptimizer
        Configured optimizer instance.
    """
    if name == "qiea":
        multi = objective_type == "multi"
        return QIEAOptimizer(
            search_space=search_space,
            n_trials=n_trials,
            population_size=20,
            multi_objective=multi,
            mo_objectives=mo_objectives or ("acc", "time"),
            seed=seed,
        )
    elif name == "random":
        multi = objective_type == "multi"
        return RandomSearchOptimizer(
            search_space,
            n_trials,
            seed=seed,
            multi_objective=multi,
            mo_objectives=mo_objectives or ("acc", "time"),
        )
    elif name == "ga":
        multi = objective_type == "multi"
        return GAOptimizer(
            search_space,
            n_trials,
            population_size=20,
            seed=seed,
            multi_objective=multi,
            mo_objectives=mo_objectives or ("acc", "time"),
        )
    elif name == "tpe":
        multi = objective_type == "multi"
        return OptunaTPEOptimizer(
            search_space,
            n_trials,
            seed=seed,
            multi_objective=multi,
            mo_objectives=mo_objectives or ("acc", "time"),
        )
    elif name == "asha":
        multi = objective_type == "multi"
        return OptunaASHAOptimizer(
            search_space,
            n_trials,
            max_resource=10,
            seed=seed,
            multi_objective=multi,
            mo_objectives=mo_objectives or ("acc", "time"),
        )
    else:
        raise ValueError(f"Unknown optimizer {name}")


def main():
    """
    CLI entry point:

    Example
    -------
    Single-objective XGBoost, QIEA:

        python run_experiment.py \\
            --optimizer qiea \\
            --task xgboost_breast_cancer \\
            --n_trials 50 \\
            --seed 0 \\
            --out_dir D:/QIEA_HPO \\
            --objective_type single

    Multi-objective (QIEA only, accuracy/time):

        python run_experiment.py \\
            --optimizer qiea \\
            --task cnn_cifar10 \\
            --n_trials 50 \\
            --seed 0 \\
            --out_dir D:/QIEA_HPO \\
            --objective_type multi
    """
    parser = argparse.ArgumentParser(description="QIEA-HPO experiments")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["qiea", "random", "ga", "tpe", "asha"],
        required=True,
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "xgboost_breast_cancer",
            "xgboost_wine",
            "svm_breast_cancer",
            "mlp_wine",
            "cnn_cifar10",
            "cnn_mnist"
        ],
        required=True,
        help="Benchmark task to run.",
    )
    parser.add_argument("--n_trials", type=int, default=50, help="Evaluation budget.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory for JSON results and logs.",
    )
    parser.add_argument(
        "--objective_type",
        type=str,
        choices=["single", "multi"],
        default="single",
        help=(
            "Objective mode. "
            "'single' = scalar accuracy. "
            "'multi' = (for now) used by QIEA for multi-objective mode; "
            "baselines remain single-objective."
        ),
    )
    parser.add_argument(
        "--mo_objectives",
        type=str,
        choices=["acc_time", "acc_epochs"],
        default="acc_time",
        help=(
            "Which objective pair to use in multi-objective mode: "
            "'acc_time' = (accuracy, training time), "
            "'acc_epochs' = (accuracy, number of epochs). "
            "Ignored if --objective_type=single."
        ),
    )
    args = parser.parse_args()

    if args.objective_type == "multi":
        if args.mo_objectives == "acc_time":
            mo_pair = ("acc", "time")
        elif args.mo_objectives == "acc_epochs":
            mo_pair = ("acc", "epochs")
        else:
            raise ValueError(f"Unknown mo_objectives {args.mo_objectives}")
    else:
        mo_pair = None  # not used

    run_name = f"{args.optimizer}_{args.task}_seed{args.seed}_{args.objective_type}"
    os.makedirs(args.out_dir, exist_ok=True)

    # Configure logger
    setup_logger(log_dir=args.out_dir, run_name=run_name)

    set_global_seed(args.seed)

    logger.info(
        f"Starting run: optimizer={args.optimizer}, task={args.task}, "
        f"n_trials={args.n_trials}, seed={args.seed}, "
        f"objective_type={args.objective_type}"
    )

    objective_fn, meta = build_objective(args.task)
    logger.info(f"Task metadata: {meta}")

    search_space = build_search_space(args.task)
    logger.info(f"Search space has {len(search_space)} hyperparameters.")

    optimizer = build_optimizer(
        name=args.optimizer,
        search_space=search_space,
        n_trials=args.n_trials,
        seed=args.seed,
        objective_type=args.objective_type,
        mo_objectives=mo_pair,
    )

    # ASHA expects an objective(params, resource) signature; in this first
    # implementation we always use max_resource but still expose the hook.
    if args.optimizer == "asha":

        def obj_with_res(params, resource):
            """Object with resources"""
            score, info = objective_fn(params)
            # Attach the resource value to the info dict for later analysis
            info = dict(info)
            info["resource"] = resource
            return score, info

        best_params = optimizer.optimize(obj_with_res)
    else:
        best_params = optimizer.optimize(objective_fn)

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best value: {optimizer.best_value}")

    run_prefix = os.path.join(args.out_dir, run_name)
    out_trials = run_prefix + "_trials.json"
    out_best = run_prefix + "_best.json"

    save_trials(optimizer.trials, out_trials)
    save_best(best_params, out_best)

    # Save Pareto front if available
    if args.objective_type == "multi" and getattr(optimizer, "archive", None) is not None:
        try:
            front = optimizer.archive.get_front()
            out_pareto = run_prefix + "_pareto.json"
            save_pareto_front(front, out_pareto)
            logger.info(f"Saved Pareto front with {len(front)} entries to {out_pareto}")
        except Exception as e:
            logger.warning(f"Failed to save Pareto front: {e}")

    logger.info("Run completed successfully.")


if __name__ == "__main__":
    main()
