"""
Global seeding utilities for reproducible experiments.

This module exposes a single helper `set_global_seed` which seeds:
- Python's `random`
- NumPy
- PyTorch (CPU and CUDA, if available)
"""

import random

import torch
import numpy as np
from loguru import logger


def set_global_seed(seed: int = 0) -> None:
    """
    Set global random seeds across common libraries.

    Parameters
    ----------
    seed : int, optional
        Seed value for random number generators (default: 0).
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")
