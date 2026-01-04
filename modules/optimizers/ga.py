"""
Genetic algorithm (GA) baseline for hyperparameter optimization.
"""

import numpy as np
from typing import Any, Callable

from loguru import logger

from .base import BaseOptimizer
from .multiobjective import ParetoArchive


class GAOptimizer(BaseOptimizer):
    """
    Genetic algorithm baseline with optional multi-objective tracking.

    Search is still driven by scalar fitness (accuracy); in multi-objective
    mode we simply maintain a Pareto archive over (acc, time) for analysis.

    - Direct encoding of hyperparameters (not Q-bits).
    - Tournament selection, uniform crossover, Gaussian / step mutation.
    - Elitism to preserve the best individual per generation.
    """

    def __init__(
            self,
            search_space: dict[str, Any],
            n_trials: int,
            population_size: int = 20,
            crossover_prob: float = 0.8,
            mutation_prob: float = 0.1,
            seed: int = 0,
            multi_objective: bool = False,
            mo_objectives: tuple[str, str] = ("acc", "time"),
    ):
        super().__init__(search_space, n_trials, seed)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)
        self.multi_objective = multi_objective
        self.mo_objectives = mo_objectives

        if multi_objective:
            obj1, obj2 = mo_objectives
            maximize = {obj1: True, obj2: False}
            self.archive = ParetoArchive(maximize)
        else:
            self.archive = None

    def _random_params(self) -> dict[str, Any]:
        """
        Sample a random individual (hyperparameter configuration).
        """
        params = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "categorical":
                params[name] = self.rng.choice(spec["choices"])
            elif spec["type"] == "int":
                params[name] = int(self.rng.integers(spec["low"], spec["high"] + 1))
            elif spec["type"] == "float":
                params[name] = float(self.rng.uniform(spec["low"], spec["high"]))
        return params

    def _mutate(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Apply mutation to a configuration with per-parameter probability.
        """
        new_params = params.copy()
        for name, spec in self.search_space.items():
            if self.rng.random() < self.mutation_prob:
                if spec["type"] == "categorical":
                    choices = spec["choices"]
                    current = new_params[name]
                    others = [c for c in choices if c != current]
                    new_params[name] = self.rng.choice(others)
                elif spec["type"] == "int":
                    step = max(1, (spec["high"] - spec["low"]) // 10)
                    new_val = new_params[name] + self.rng.integers(-step, step + 1)
                    new_params[name] = int(np.clip(new_val, spec["low"], spec["high"]))
                elif spec["type"] == "float":
                    span = spec["high"] - spec["low"]
                    new_val = new_params[name] + self.rng.normal(0, span * 0.1)
                    new_params[name] = float(np.clip(new_val, spec["low"], spec["high"]))
        return new_params

    def _crossover(
            self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Uniform crossover between two parents at the hyperparameter level.
        """
        child = {}
        for name in self.search_space.keys():
            if self.rng.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        return child

    def _tournament_select(
            self,
            pop: list[dict[str, Any]],
            fitness: list[float],
            k: int = 3,
    ) -> dict[str, Any]:
        """
        Tournament selection over the population.

        Parameters
        ----------
        pop : list of dict
            Current population.
        fitness : list of float
            Fitness values aligned with `pop`.
        k : int, optional
            Tournament size, default 3.

        Returns
        -------
        dict
            Selected parent individual.
        """
        idxs = self.rng.choice(len(pop), size=k, replace=False)
        best_idx = idxs[np.argmax([fitness[i] for i in idxs])]
        return pop[best_idx]

    def optimize(
            self,
            objective: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run GA-based HPO until the evaluation budget (n_trials) is exhausted.

        Parameters
        ----------
        objective : callable
            Objective function mapping hyperparameters dict -> (score, info).

        Returns
        -------
        dict
            Best hyperparameter configuration found.
        """
        logger.info(
            f"Starting GA optimization with population_size={self.population_size}, "
            f"n_trials={self.n_trials}"
        )

        # Initial population
        population = [self._random_params() for _ in range(self.population_size)]
        fitness: list[float] = []
        infos: list[dict[str, Any]] = []
        best_val = -np.inf
        best_params = None
        n_evals = 0

        def eval_pop(pop):
            """Evaluate the population."""
            nonlocal best_val, best_params, n_evals
            fitness.clear()
            infos.clear()
            for p in pop:
                score, info = objective(p)
                n_evals += 1

                if self.multi_objective and self.archive is not None:
                    obj1, obj2 = self.mo_objectives
                    val1 = info.get(obj1, score)
                    val2 = info.get(obj2, 0.0)
                    self.archive.update(p, {obj1: val1, obj2: val2})

                self.trials.append({"params": p, "score": score, "info": info})
                fitness.append(score)
                infos.append(info)
                if score > best_val:
                    best_val = score
                    best_params = p
                    logger.debug(
                        f"[Eval {n_evals}] New best score={best_val:.4f} with {p}"
                    )
                if n_evals >= self.n_trials:
                    return True
            return False

        if eval_pop(population):
            self.best_params = best_params
            self.best_value = best_val
            logger.info(f"GA finished after initial population. Best={best_val:.4f}")
            return best_params

        # Generations
        gen = 0
        while n_evals < self.n_trials:
            gen += 1
            logger.info(
                f"Starting GA generation {gen} "
                f"(evaluations so far: {n_evals}/{self.n_trials})"
            )
            new_population: list[dict[str, Any]] = []
            # Elitism: keep the best
            elite_idx = int(np.argmax(fitness))
            new_population.append(population[elite_idx])

            while len(new_population) < self.population_size:
                # selection
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                # crossover
                if self.rng.random() < self.crossover_prob:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                # mutation
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            if eval_pop(population):
                break

        self.best_params = best_params
        self.best_value = best_val
        logger.info(
            f"GA optimization finished after {n_evals} evaluations. "
            f"Best score={best_val:.4f}"
        )
        return best_params
