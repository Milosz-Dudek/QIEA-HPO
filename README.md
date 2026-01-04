# QIEA-HPO

Quantum-Inspired Evolutionary Algorithm for Hyperparameter Optimization (QIEA-HPO) is a
research-oriented playground for comparing a quantum-inspired optimizer against several
baselines on both tabular and vision benchmarks. The repository provides minimal wiring to
run individual experiments or full sweeps across optimizers, tasks, and random seeds while
saving rich JSON logs for later analysis.

## Project layout

- `run_experiment.py`: CLI for a single experiment. It wires an optimizer, task-specific
  objective, search space, and logging. Supports single- and multi-objective modes.
- `run_all_experiments.py`: Launcher that executes the Cartesian product of optimizers,
  tasks, seeds, and objective modes by calling `run_experiment.py` as a subprocess.
- `modules/optimizers`: Optimizer implementations (`QIEAOptimizer`, random search, genetic
  algorithm, Optuna TPE, Optuna ASHA) plus the Pareto archive helper.
- `modules/tasks`: Objective builders for tabular datasets (XGBoost, SVM, MLP) and vision
  datasets (CIFAR-10, MNIST).
- `modules/utils`: Shared helpers for search spaces, reproducible seeding, JSON logging,
  and logging configuration.

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

2. Install Python dependencies. The code expects `numpy`, `torch`, `torchvision`, `scikit-learn`,
   `xgboost`, `optuna`, and `loguru` at minimum. A quick install for typical CPU-only runs:

   ```bash
   pip install numpy torch torchvision scikit-learn xgboost optuna loguru
   ```

   Adjust versions or CUDA builds as needed for your hardware and research setup.

## Running a single experiment

Use `run_experiment.py` to configure one optimizer/task combination:

```bash
python run_experiment.py \
    --optimizer qiea \
    --task xgboost_breast_cancer \
    --n_trials 50 \
    --seed 0 \
    --out_dir results \
    --objective_type single
```

Key flags:

- `--optimizer`: `qiea`, `random`, `ga`, `tpe`, or `asha`.
- `--task`: one of `xgboost_breast_cancer`, `xgboost_wine`, `svm_breast_cancer`,
  `mlp_wine`, `cnn_cifar10`, or `cnn_mnist`.
- `--objective_type`: `single` for scalar accuracy, `multi` to enable multi-objective
  QIEA (optimizing accuracy vs. time or epochs).
- `--mo_objectives`: choose `acc_time` (accuracy/time) or `acc_epochs` when
  `--objective_type multi`.

Outputs are written under `--out_dir` with three JSON files per run:

- `<run>_trials.json`: list of evaluated hyperparameters and metrics.
- `<run>_best.json`: best hyperparameters and score (scalar accuracy) for quick lookup.
- `<run>_pareto.json`: Pareto front entries when multi-objective mode is active and
  available.

## Running full sweeps

`run_all_experiments.py` spawns the full grid of optimizers × tasks × seeds × objective
modes. Example:

```bash
python run_all_experiments.py --n_trials 50 --seeds 0,1,2 --out_dir results
```

The launcher uses `run_experiment.py` internally and logs a concise status line per run.
Comment or extend the `optimizers` and `tasks` lists in the script to customize the grid
for your research scenario.

## Research notes

- **Search spaces** are defined in `modules/utils/search_space.py` and kept intentionally
  lightweight for quick experimentation. Adjust ranges or bin counts to suit new studies.
- **Objectives** returned by functions in `modules/tasks` follow a common interface:
  `objective(hparams) -> (score, info_dict)`, where `score` is validation accuracy in `[0, 1]`
  and `info_dict` at least contains timing metadata. This design simplifies swapping
  optimizers without touching task code.
- **Multi-objective QIEA** stores a Pareto archive (`ParetoArchive`) and still exposes a
  scalar `best_value` for compatibility with baseline logging. Downstream analysis can
  inspect the saved Pareto front for trade-off exploration.
- **Reproducibility**: `modules.utils.seed.set_global_seed` sets Python, NumPy, and PyTorch
  seeds. Ensure deterministic CuDNN settings or dataset downloads according to your
  environment needs.

## Validation and troubleshooting

- The repository is intentionally minimal and does not ship with an automated test suite.
  When adapting the code, run a short sanity check such as one or two trials on
  `xgboost_breast_cancer` to verify your Python environment and GPU/CPU settings.
- Review the logging output (Loguru) in `--out_dir` if a run fails; most exceptions are
  surfaced directly from the task or optimizer implementations.

## Module exports

`modules/__init__.py` now provides convenient imports for common components:

```python
from modules import (
    QIEAOptimizer, RandomSearchOptimizer, GAOptimizer,
    OptunaTPEOptimizer, OptunaASHAOptimizer, ParetoArchive,
    get_search_space, setup_logger, set_global_seed,
    make_xgboost_objective, make_svm_objective, make_mlp_objective,
    make_cifar10_objective, make_mnist_objective,
)
```

This is helpful for notebooks or interactive exploration where brevity matters.

