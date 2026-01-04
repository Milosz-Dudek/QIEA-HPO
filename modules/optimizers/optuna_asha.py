"""
Optuna-based ASHA-style (Hyperband-like) optimizer wrapper.
"""
from typing import Any, Callable

import optuna
from loguru import logger

from .base import BaseOptimizer
from .multiobjective import ParetoArchive


class OptunaASHAOptimizer(BaseOptimizer):
    """
    ASHA / Hyperband-like multi-fidelity optimization via Optuna.

    We approximate ASHA with Optuna's HyperbandPruner (widely available).
    Search remains single-objective (accuracy), but in multi-objective
    mode we record (acc, time) in a Pareto archive.
    """

    def __init__(
            self,
            search_space: dict[str, Any],
            n_trials: int,
            max_resource: int = 50,
            min_resource: int = 5,
            reduction_factor: int = 3,
            seed: int = 0,
            multi_objective: bool = False,
            mo_objectives: tuple[str, str] = ("acc", "time"),
    ):
        super().__init__(search_space, n_trials, seed)
        self.max_resource = max_resource
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.multi_objective = multi_objective
        self.mo_objectives = mo_objectives
        if multi_objective:
            obj1, obj2 = mo_objectives
            maximize = {obj1: True, obj2: False}
            self.archive = ParetoArchive(maximize)
        else:
            self.archive = None

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
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
            objective_with_resource: Callable[[dict[str, Any], int], tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run ASHA-style optimization via Optuna's SuccessiveHalvingPruner.

        Parameters
        ----------
        objective_with_resource : callable
            Objective taking (params, resource) and returning (score, info).

        Returns
        -------
        dict
            Best hyperparameter configuration found.
        """
        logger.info(
            "Starting Optuna ASHA-style optimization "
            f"(max_resource={self.max_resource}, n_trials={self.n_trials})."
        )

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            reduction_factor=self.reduction_factor,
            max_resource=self.max_resource,
        )
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        def optuna_objective(trial: optuna.Trial) -> float:
            """
            In a full ASHA integration, your inner training loop would
            call trial.report() + trial.should_prune() for intermediate
            results. Right now, we still pass max_resource and let
            pruning be "available" for future extensions.
            """
            params = self._suggest_params(trial)
            resource = self.max_resource  # simple version
            score, info = objective_with_resource(params, resource)
            if self.multi_objective and self.archive is not None:
                obj1, obj2 = self.mo_objectives
                val1 = info.get(obj1, score)
                val2 = info.get(obj2, 0.0)
                self.archive.update(params, {obj1: val1, obj2: val2})

            self.trials.append(
                {"params": params, "score": score, "info": info, "resource": resource}
            )
            return score

        study.optimize(optuna_objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_value = study.best_value
        logger.info(
            f"Optuna ASHA-style optimization finished. Best score={study.best_value:.4f}"
        )
        return study.best_params
