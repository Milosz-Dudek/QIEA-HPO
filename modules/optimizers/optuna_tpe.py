"""
Optuna-based TPE (Bayesian optimization) optimizer wrapper.
"""

from typing import Any, Callable

import optuna
from loguru import logger

from .base import BaseOptimizer
from .multiobjective import ParetoArchive


class OptunaTPEOptimizer(BaseOptimizer):
    """
    TPE-based Bayesian optimization via Optuna.

    Single-objective search on accuracy, optional multi-objective tracking
    of (acc, time) via a Pareto archive.
    """

    def __init__(
            self,
            search_space: dict[str, Any],
            n_trials: int,
            seed: int = 0,
            multi_objective: bool = False,
            mo_objectives: tuple[str, str] = ("acc", "time"),
    ):
        super().__init__(search_space, n_trials, seed)
        self.multi_objective = multi_objective
        self.mo_objectives = mo_objectives
        self.archive = (
            ParetoArchive({"acc": True, "time": False})
            if multi_objective
            else None
        )

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest a configuration from the search space using Optuna's Trial.
        """
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
        return params

    def optimize(
            self,
            objective: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run TPE-based optimization for the given objective.

        Parameters
        ----------
        objective : callable
            Objective function mapping hyperparameters dict -> (score, info).

        Returns
        -------
        dict
            Best hyperparameter configuration found.
        """
        logger.info(f"Starting Optuna TPE optimization for {self.n_trials} trials.")

        def optuna_objective(trial: optuna.Trial) -> float:
            """Optuna objective - single objective"""
            params = self._suggest_params(trial)
            score, info = objective(params)

            if self.multi_objective and self.archive is not None:
                acc_name, time_name = self.mo_objectives
                acc = info.get(acc_name, score)
                t = info.get(time_name, 0.0)
                self.archive.update(params, {"acc": acc, "time": t})

            self.trials.append({"params": params, "score": score, "info": info})
            return score

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_value = study.best_value
        logger.info(f"Optuna TPE finished. Best score={self.best_value:.4f}")
        return study.best_params
