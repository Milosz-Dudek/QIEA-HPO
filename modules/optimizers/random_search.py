"""
Random search optimizer for hyperparameter optimization.
"""

import numpy as np
from typing import Any, Callable

from loguru import logger

from .base import BaseOptimizer
from .multiobjective import ParetoArchive


class RandomSearchOptimizer(BaseOptimizer):
    """
    Simple random search baseline with optional multi-objective logging.
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
        self.rng = np.random.default_rng(seed)
        self.multi_objective = multi_objective
        self.mo_objectives = mo_objectives

        if multi_objective:
            obj1, obj2 = mo_objectives
            maximize = {obj1: True, obj2: False}
            self.archive = ParetoArchive(maximize)
        else:
            self.archive = None

    def _sample_params(self) -> dict[str, Any]:
        """
        Draw a random configuration from the search space.
        """
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "categorical":
                params[name] = self.rng.choice(spec["choices"])
            elif spec["type"] == "int":
                params[name] = int(self.rng.integers(spec["low"], spec["high"] + 1))
            elif spec["type"] == "float":
                params[name] = float(
                    self.rng.uniform(spec["low"], spec["high"])
                )
            else:
                raise ValueError(f"Unknown type {spec['type']}")
        return params

    def optimize(
            self,
            objective: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run random search for a fixed number of trials.

        Parameters
        ----------
        objective : callable
            Objective function mapping hyperparameters dict -> (score, info).

        Returns
        -------
        dict
            Best hyperparameter configuration found.
        """
        logger.info(f"Starting RandomSearch for {self.n_trials} trials.")
        best_val = -np.inf
        best_params = None
        for i in range(self.n_trials):
            params = self._sample_params()
            score, info = objective(params)

            if self.multi_objective and self.archive is not None:
                obj1, obj2 = self.mo_objectives  # e.g. ("acc", "time") or ("acc", "epochs")
                # First objective: default to `score` if not present in info
                val1 = info.get(obj1, score)
                # Second objective: require it in info, fall back to 0.0 if missing
                val2 = info.get(obj2, 0.0)
                self.archive.update(params, {obj1: val1, obj2: val2})

            self.trials.append({"params": params, "score": score, "info": info})
            if score > best_val:
                best_val = score
                best_params = params

        self.best_params = best_params
        self.best_value = best_val
        logger.info(f"RandomSearch finished. Best score={best_val:.4f}")
        return best_params
