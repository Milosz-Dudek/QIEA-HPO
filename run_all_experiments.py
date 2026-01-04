"""
Batch launcher for running all (optimizer, task, seed) combinations.

This script calls `run_experiment.py` as a subprocess for each combination
and stores all outputs under the specified output directory.

Typical usage:

    python run_all_experiments.py \\
        --n_trials 50 \\
        --seeds 0,1,2 \\
        --out_dir D:/QIEA_HPO \\
        --objective_type single
"""

import argparse
import subprocess
import itertools
import os
import sys
from pathlib import Path
from typing import List

from loguru import logger


def run_one(
        optimizer: str,
        task: str,
        n_trials: int,
        seed: int,
        out_dir: str,
        objective_type: str,
        mo_objectives: str,
):
    """
    Run a single (optimizer, task, seed) combination by invoking run_experiment.py.

    Parameters
    ----------
    optimizer : str
        Optimizer name, e.g. "qiea", "random", "ga", "tpe", "asha".
    task : str
        Task name, e.g. "xgboost_breast_cancer".
    n_trials : int
        Number of HPO trials to run.
    seed : int
        Random seed for this run.
    out_dir : str
        Output directory (passed through to run_experiment.py).
    objective_type : {"single", "multi"}
        Objective mode. Currently only QIEA consumes "multi" internally;
        baselines remain single-objective.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_experiment.py")),
        "--optimizer", optimizer,
        "--task", task,
        "--n_trials", str(n_trials),
        "--seed", str(seed),
        "--out_dir", out_dir,
        "--objective_type", objective_type,
        "--mo_objectives", mo_objectives,
    ]
    logger.info(f"[LAUNCH] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning(
            f"[FAILED] optimizer={optimizer}, task={task}, seed={seed}, "
            f"objective_type={objective_type}, returncode={result.returncode}"
        )
    else:
        logger.info(
            f"[DONE] optimizer={optimizer}, task={task}, seed={seed}, "
            f"objective_type={objective_type}"
        )


def parse_seeds(seeds_str: str) -> List[int]:
    """
    Parse seeds argument like '0' or '0,1,2' into a list of ints.

    Parameters
    ----------
    seeds_str : str
        Comma-separated seeds, e.g. "0" or "0,1,2".

    Returns
    -------
    list[int]
        Parsed integer seeds.
    """
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip() != ""]


def main():
    """
    CLI entry point to launch all experiments.

    The Cartesian product:
        seeds × optimizers × tasks
    is executed sequentially by default.
    """
    parser = argparse.ArgumentParser(
        description="Run all optimizer/task combinations for QIEA-HPO"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of trials per optimizer/task/seed",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated list of seeds, e.g. '0' or '0,1,2'",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="D:/QIEA_HPO",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default="python",
        help="Python executable to use (e.g. 'python', 'python3', 'py')",
    )
    parser.add_argument(
        "--mo_objectives",
        type=str,
        choices=["acc_time", "acc_epochs"],
        default="acc_time",
        help="Forwarded to run_experiment.py in multi-objective mode.",
    )
    args = parser.parse_args()

    optimizers = [
        "random",
        "qiea",
        "ga",
        "tpe",
        "asha"
    ]
    tasks = [
        "xgboost_breast_cancer",
        "xgboost_wine",
        "svm_breast_cancer",
        "mlp_wine",
        # "cnn_cifar10",
        "cnn_mnist"
    ]
    objective_types = ["single", "multi"]
    seeds = parse_seeds(args.seeds)

    os.makedirs(args.out_dir, exist_ok=True)

    # Simple loguru configuration for the launcher itself
    from modules.utils.logging_utils import setup_logger

    setup_logger(log_dir=args.out_dir, run_name="launcher")

    logger.info("=== Running all experiments ===")
    logger.info(f"Optimizers:      {optimizers}")
    logger.info(f"Tasks:           {tasks}")
    logger.info(f"Seeds:           {seeds}")
    logger.info(f"Objective Types: {objective_types}")
    logger.info(f"n_trials:        {args.n_trials}")
    logger.info(f"out_dir:         {args.out_dir}")

    for seed, optimizer, task, objective_type in itertools.product(seeds, optimizers, tasks, objective_types):
        run_one(
            optimizer=optimizer,
            task=task,
            n_trials=args.n_trials,
            seed=seed,
            out_dir=args.out_dir,
            objective_type=objective_type,
            mo_objectives=args.mo_objectives,
        )

    logger.info("=== All experiments attempted ===")


if __name__ == "__main__":
    main()
