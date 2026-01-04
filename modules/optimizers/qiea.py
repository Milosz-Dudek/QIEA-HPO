"""
Quantum-inspired evolutionary algorithm (QIEA) optimizers for HPO.

This module implements:
- A single-objective QIEA optimizer (accuracy as scalar target).
- A multi-objective variant that maintains a Pareto archive over
  (accuracy, time) and uses non-dominated solutions as guides.
"""

import numpy as np
from typing import Any, Callable, Tuple, Optional

from loguru import logger

from .base import BaseOptimizer
from .multiobjective import ParetoArchive


class QIEAOptimizer(BaseOptimizer):
    """
    Quantum-Inspired Evolutionary Algorithm for HPO.

    Single-objective mode:
        - maximizes scalar score (accuracy) using Q-bit encoding.

    Multi-objective mode:
        - still tracks scalar accuracy as `best_value` for compatibility,
          but maintains a Pareto archive over (acc, time) and uses archive
          members as rotation guides.
    """

    def __init__(
            self,
            search_space: dict[str, Any],
            n_trials: int,
            population_size: int = 20,
            max_generations: Optional[int] = None,
            base_rotation: float = 0.05,
            stagnation_generations: int = 10,
            seed: int = 0,
            multi_objective: bool = False,
            mo_objectives: Tuple[str, str] = ("acc", "time"),
    ):
        super().__init__(search_space, n_trials, seed)
        self.population_size = population_size
        self.max_generations = max_generations
        self.base_rotation = base_rotation
        self.stagnation_generations = stagnation_generations
        self.rng = np.random.default_rng(seed)

        self.multi_objective = multi_objective
        self.mo_objectives = mo_objectives
        if multi_objective:
            obj1, obj2 = mo_objectives
            maximize = {obj1: True, obj2: False}
            self.archive = ParetoArchive(maximize)
        else:
            self.archive = None

        self._build_encoding()

    # Encoding
    def _build_encoding(self) -> None:
        """
        Build a binary encoding for each hyperparameter.

        Encoding rules
        --------------
        - categorical: index in the choices list encoded in binary
        - int: integer in [low, high] encoded in binary
        - float: discretized into bins in [low, high], index encoded
        """
        self.param_names = []
        self.param_schemas = {}  # Name -> dict describing type & levels
        self.total_bits = 0

        for name, spec in self.search_space.items():
            self.param_names.append(name)
            ptype = spec["type"]

            if ptype == "categorical":
                choices = spec["choices"]
                n_levels = len(choices)
                n_bits = int(np.ceil(np.log2(n_levels)))
                self.param_schemas[name] = {
                    "type": ptype,
                    "choices": choices,
                    "n_bits": n_bits,
                    "n_levels": n_levels,
                }
            elif ptype == "int":
                low, high = spec["low"], spec["high"]
                n_levels = high - low + 1
                n_bits = int(np.ceil(np.log2(n_levels)))
                self.param_schemas[name] = {
                    "type": ptype,
                    "low": low,
                    "high": high,
                    "n_bits": n_bits,
                    "n_levels": n_levels,
                }
            elif ptype == "float":
                low, high = spec["low"], spec["high"]
                n_bins = spec.get("n_bins", 16)
                n_bits = int(np.ceil(np.log2(n_bins)))
                self.param_schemas[name] = {
                    "type": ptype,
                    "low": low,
                    "high": high,
                    "n_bins": n_bins,
                    "n_bits": n_bits,
                }
            else:
                raise ValueError(f"Unknown param type: {ptype}")

            self.param_schemas[name]["start_bit"] = self.total_bits
            self.total_bits += self.param_schemas[name]["n_bits"]

        # Q-population: each individual is [alpha, beta] for each bit
        # shape: (population_size, total_bits, 2)
        self.population = np.zeros(
            (self.population_size, self.total_bits, 2), dtype=np.float64
        )
        # Init in equal superposition
        self.population[:, :, 0] = 1 / np.sqrt(2)
        self.population[:, :, 1] = 1 / np.sqrt(2)
        logger.info(
            f"QIEA encoding built. total_bits={self.total_bits}, "
            f"population_size={self.population_size}"
        )

    # Sampling & decoding
    def _sample_individual(self, q_individual: np.ndarray) -> dict[str, Any]:
        """
        Observe a quantum individual to produce concrete hyperparameters.

        Parameters
        ----------
        q_individual : np.ndarray
            Amplitudes array of shape (total_bits, 2) holding [alpha, beta].

        Returns
        -------
        dict
            Decoded hyperparameters.
        """
        bits = self.rng.random(self.total_bits) < np.square(q_individual[:, 1])
        params = {}
        for name in self.param_names:
            schema = self.param_schemas[name]
            start = schema["start_bit"]
            n_bits = schema["n_bits"]
            code = bits[start: start + n_bits]
            val = self._decode_bits(name, code)
            params[name] = val
        return params

    def _decode_bits(self, name: str, bits: np.ndarray) -> Any:
        """
        Convert a bit segment into a parameter value.
        """
        schema = self.param_schemas[name]
        ptype = schema["type"]
        idx = 0
        for i, b in enumerate(bits):
            if b:
                idx += 1 << i
        if ptype == "categorical":
            idx = min(idx, schema["n_levels"] - 1)
            return schema["choices"][idx]
        elif ptype == "int":
            idx = min(idx, schema["n_levels"] - 1)
            return schema["low"] + idx
        elif ptype == "float":
            idx = min(idx, schema["n_bins"] - 1)
            frac = idx / max(1, schema["n_bins"] - 1)
            return schema["low"] + frac * (schema["high"] - schema["low"])
        else:
            raise ValueError

    # Rotation
    def _rotate_towards(
            self,
            q_individual: np.ndarray,
            target_bits: np.ndarray,
            theta: float,
    ) -> np.ndarray:
        """
        Rotate each bit's amplitudes toward target_bits (0 or 1).

        The rotation increases |beta|^2 when target bit is 1, or |alpha|^2
        when target bit is 0, with an angle `theta` (possibly generation-dependent).
        """
        new_q = q_individual.copy()
        for i in range(self.total_bits):
            alpha, beta = new_q[i]
            # if target=1 we want |beta|^2 to increase; if target=0, increase |alpha|^2
            if target_bits[i] == 1:
                rot = np.array(
                    [[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]
                )
            else:
                rot = np.array(
                    [[np.cos(-theta), -np.sin(-theta)],
                     [np.sin(-theta), np.cos(-theta)]]
                )
            alpha, beta = rot @ np.array([alpha, beta])
            norm = np.sqrt(alpha ** 2 + beta ** 2)
            new_q[i, 0] = alpha / norm
            new_q[i, 1] = beta / norm
        return new_q

    def _params_to_bits(self, params: dict[str, Any]) -> np.ndarray:
        """
        Encode concrete hyperparameters back into a bitstring.
        """
        bits = np.zeros(self.total_bits, dtype=np.int8)
        for name in self.param_names:
            schema = self.param_schemas[name]
            start = schema["start_bit"]
            n_bits = schema["n_bits"]
            # Figure index
            if schema["type"] == "categorical":
                idx = schema["choices"].index(params[name])
            elif schema["type"] == "int":
                idx = params[name] - schema["low"]
            else:  # float
                frac = (params[name] - schema["low"]) / (
                        schema["high"] - schema["low"]
                )
                idx = int(round(frac * (schema["n_bins"] - 1)))
            idx = max(0, min(idx, (1 << n_bits) - 1))
            for i in range(n_bits):
                if idx & (1 << i):
                    bits[start + i] = 1
        return bits

    def _population_entropy(self, q_population: np.ndarray) -> float:
        """
        Estimate bit-level diversity via average Bernoulli entropy across the population.

        q_population: shape (pop_size, total_bits, 2) with amplitudes [alpha, beta].
        Returns H in [0, 1], where 1 ~ high diversity, 0 ~ fully collapsed.
        """
        # Probability of bit=1 for each bit across population
        probs_1 = (q_population[:, :, 1] ** 2).mean(axis=0)  # shape (total_bits,)
        eps = 1e-12
        p = np.clip(probs_1, eps, 1.0 - eps)
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        # Normalize: max entropy of Bernoulli is 1 bit
        return float(entropy.mean())

    def optimize(
            self,
            objective: Callable[[dict[str, Any]], Tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run QIEA-based hyperparameter optimization.

        In single-objective mode, `objective` is assumed to return:
            score = accuracy (maximization)
        and `info` is arbitrary metadata.

        In multi-objective mode, `objective` is still expected to return
        a scalar score (typically accuracy), but `info` must contain:
            info[obj1]  -> first objective (e.g. "acc")
            info[obj2]  -> second objective (e.g. "time" or "epochs")
        which are used to maintain the Pareto archive and guide Q-bit rotations.
        """
        mode = "multi-objective" if self.multi_objective else "single-objective"
        logger.info(
            f"Starting QIEA optimize in {mode} mode "
            f"(n_trials={self.n_trials}, pop_size={self.population_size})"
        )

        n_evals = 0
        best_val = -np.inf
        best_params = None
        no_improve_gens = 0

        max_gens = (
            self.max_generations
            if self.max_generations is not None
            else max(1, self.n_trials // self.population_size)
        )

        for gen in range(max_gens):
            logger.info(
                f"[QIEA] Generation {gen + 1}/{max_gens} "
                f"(evaluations so far: {n_evals}/{self.n_trials})"
            )
            samples: list[dict[str, Any]] = []
            fitness: list[float] = []
            infos: list[dict[str, Any]] = []

            # Evaluate current population
            for p_idx in range(self.population_size):
                params = self._sample_individual(self.population[p_idx])
                score, info = objective(params)
                n_evals += 1

                # Update Pareto archive in multi-objective mode
                if self.multi_objective and self.archive is not None:
                    obj1, obj2 = self.mo_objectives
                    val1 = info.get(obj1, score)  # usually acc
                    val2 = info.get(obj2, 0.0)  # time or epochs
                    self.archive.update(params, {obj1: val1, obj2: val2})

                # Log trial
                self.trials.append(
                    {"params": params, "score": score, "info": info, "gen": gen}
                )
                samples.append(params)
                fitness.append(score)
                infos.append(info)

                # Track global best (single scalar: score)
                if score > best_val:
                    best_val = score
                    best_params = params
                    best_info = info
                    no_improve_gens = 0
                    logger.debug(
                        f"[QIEA] New best score={best_val:.4f} at eval={n_evals} "
                        f"with info={best_info}"
                    )

                # Hard budget stop
                if n_evals >= self.n_trials:
                    self.best_params = best_params
                    self.best_value = best_val
                    return best_params

            # End of generation: update Q-population
            no_improve_gens += 1

            # Choose guide for rotation
            if self.multi_objective and self.archive is not None and self.archive.entries:
                # Multi-objective: guide from Pareto front (random member for now)
                idx = self.rng.integers(0, len(self.archive.entries))
                guide_params = self.archive.entries[idx]["params"]
            else:
                # Single-objective (or empty archive): use global best if available,
                # otherwise best in this generation.
                if best_params is not None:
                    guide_params = best_params
                else:
                    gen_best_idx = int(np.argmax(fitness))
                    guide_params = samples[gen_best_idx]

            target_bits = self._params_to_bits(guide_params)

            # Diversity-adaptive rotation
            # Global progress in [0, 1]
            progress = n_evals / float(self.n_trials)

            # Estimate population diversity H in [0, 1]
            diversity = self._population_entropy(self.population)

            # Diversity-adaptive rotation:
            # - early on and/or when diversity is high -> larger angles (exploration)
            # - later and/or when diversity is low     -> smaller angles (exploitation)
            diversity_scale = 0.5 + 0.5 * diversity  # in [0.5, 1.0]
            progress_scale = 1.0 - 0.5 * progress  # in [0.5, 1.0]

            theta = self.base_rotation * diversity_scale * progress_scale

            logger.debug(
                f"[QIEA] Gen {gen + 1}: diversity={diversity:.3f}, "
                f"theta={theta:.5f}, progress={progress:.3f}, "
                f"n_evals={n_evals}/{self.n_trials}"
            )

            # Rotate all individuals toward the guide
            for p_idx in range(self.population_size):
                self.population[p_idx] = self._rotate_towards(
                    self.population[p_idx], target_bits, theta
                )

            # Catastrophe / restart mechanism
            if no_improve_gens >= self.stagnation_generations:
                half = self.population_size // 2
                self.population[:half, :, 0] = 1 / np.sqrt(2)
                self.population[:half, :, 1] = 1 / np.sqrt(2)
                no_improve_gens = 0
                logger.info(
                    "[QIEA] Catastrophe triggered: re-initialized half "
                    "of the population to uniform superposition."
                )

        # Finished max_gens (budget might not be fully used if pop_size * max_gens > n_trials)
        self.best_params = best_params
        self.best_value = best_val

        archive_size = (
            len(self.archive.entries)
            if getattr(self, "archive", None) is not None
            else 0
        )

        logger.info(
            f"QIEA finished. Best score={self.best_value:.4f}, "
            f"archive size={archive_size}"
        )

        return best_params
