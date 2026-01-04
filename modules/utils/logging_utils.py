"""
Utility helpers for logging and saving experiment outputs.

This module provides:
- JSON writers for trial histories and best configurations.
- A convenience function to configure the global loguru logger
  with console and optional file sinks.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger


def setup_logger(
        log_dir: str,
        run_name: str,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
) -> None:
    """
    Configure the global loguru logger.

    Parameters
    ----------
    log_dir : str
        Directory where a run-specific log file will be placed.
    run_name : str
        Name of the run, used to name the log file.
    console_level : str, optional
        Logging level for the console sink (default: "INFO").
    file_level : str, optional
        Logging level for the file sink (default: "DEBUG").
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / f"{run_name}.log"

    # Reset any previous configuration (important when reusing in multiple scripts)
    logger.remove()

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=console_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )

    # File sink
    logger.add(
        str(log_path),
        level=file_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention=10,
    )

    logger.info(f"Logger initialized. Log file: {log_path}")


def _to_serializable(obj: Any) -> Any:
    """
    Recursively convert common non-JSON-serializable types
    (NumPy scalars, arrays, tensors, sets, etc.) to plain
    Python types that `json.dump` can handle.
    """
    # Basic primitives are fine
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # dict
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    # NumPy scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # NumPy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Torch tensor (if available)
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Set -> list
    if isinstance(obj, set):
        return [_to_serializable(v) for v in obj]

    # Fallback: try direct JSON dump, otherwise string
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_trials(trials: list[dict[str, Any]], out_path: str) -> None:
    """
    Save a list of trial dicts to JSON, coercing types to be JSON-safe.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _to_serializable(trials)
    with path.open("w") as f:
        json.dump(clean, f, indent=2)


def save_best(best_params: dict[str, Any], out_path: str) -> None:
    """
    Save best hyperparameters to JSON, coercing types to be JSON-safe.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _to_serializable(best_params)
    with path.open("w") as f:
        json.dump(clean, f, indent=2)


def save_pareto_front(front_entries: list[dict[str, Any]], path: str) -> None:
    """
    Save Pareto front entries (list of dicts with 'params' and 'objs') as JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _to_serializable(front_entries)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved Pareto front with {len(front_entries)} entries to {path}")
