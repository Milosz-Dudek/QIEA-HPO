"""Base script for optimizers"""
import abc
from typing import Callable, Any, Optional


class BaseOptimizer(abc.ABC):
    """
    Abstract base class for HPO optimizers.
    """

    def __init__(
            self,
            search_space: dict[str, Any],
            n_trials: int,
            seed: int = 0,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed

        # Storage for all evaluated trials
        self.trials: list[dict[str, Any]] = []

        # Best single-objective result (validation accuracy)
        self.best_params: Optional[dict[str, Any]] = None
        self.best_value: Optional[float] = None

        # Optional multi-objective archive (Pareto front)
        # Optimizers that support multi-objective will populate this.
        self.archive = None

    @abc.abstractmethod
    def optimize(
            self,
            objective: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run optimization and return best hyperparameters.
        objective(hparams) -> (score, extra_info)
        """
        raise NotImplementedError
