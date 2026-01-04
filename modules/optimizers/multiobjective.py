"""
Simple Pareto archive for multi-objective evaluation.

We keep a list of non-dominated solutions, each with:
- params:  hyperparameter configuration (dict)
- objs:    objective values, e.g. {"acc": float, "time": float}

Directions (maximize/minimize) are controlled via the `maximize` mapping.
"""

from typing import Any


class ParetoArchive:
    """
    Maintain a set of non-dominated solutions.

    Each entry is a dict:
    {
        "params": dict of hyperparameters,
        "objs":   {"acc": float, "time": float, ...}
    }
    """

    def __init__(self, maximize: dict[str, bool]):
        """
        Parameters
        ----------
        maximize : dict
            Mapping objective name -> True (maximize) / False (minimize).
        """
        self.maximize = maximize
        self.entries: list[dict[str, Any]] = []

    def _dominates(self, a: dict[str, float], b: dict[str, float]) -> bool:
        """
        Return True if objective vector `a` dominates `b`.
        """
        better_or_equal_all = True
        strictly_better = False

        for name, is_max in self.maximize.items():
            av = a[name]
            bv = b[name]
            if is_max:
                if av < bv:
                    return False
                if av > bv:
                    strictly_better = True
            else:  # minimize
                if av > bv:
                    return False
                if av < bv:
                    strictly_better = True

        return better_or_equal_all and strictly_better

    def update(self, params: dict[str, Any], objs: dict[str, float]) -> None:
        """
        Update archive with a new candidate.

        - If candidate is dominated by any existing entry: discard it.
        - Else: remove entries dominated by candidate and add it.
        """
        # If dominated by any existing entry, ignore
        for entry in self.entries:
            if self._dominates(entry["objs"], objs):
                return

        # Remove dominated entries
        new_entries = []
        for entry in self.entries:
            if not self._dominates(objs, entry["objs"]):
                new_entries.append(entry)
        self.entries = new_entries

        # Add candidate
        self.entries.append({"params": params, "objs": objs})

    def get_front(self) -> list[dict[str, Any]]:
        """
        Return the current Pareto front as a list of entries.

        Each entry has the form:
            {"params": <dict>, "objs": <dict of objectives>}

        This is mainly a convenience wrapper so that callers don't need
        to know the internal attribute name.
        """
        return list(self.entries)
