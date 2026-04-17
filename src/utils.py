"""Operational utilities for directory setup, logging, and runtime monitoring.

This module provides shared helpers used across model training and inference
pipelines in the TSA project.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import torch

try:
    from src.config import Config
except ImportError:
    from src import config as _config

    class Config:  # type: ignore[too-many-instance-attributes]
        """Compatibility layer when src.config exposes module-level constants."""

        DATA_DIR: Path = Path(getattr(_config, "DATA_DIR", Path("data")))
        MODEL_DIR: Path = Path(getattr(_config, "MODEL_DIR", Path("models")))
        OUTPUT_DIR: Path = Path(getattr(_config, "OUTPUT_DIR", Path("outputs")))
        XGBOOST_OUTPUT_DIR: Path = Path(
            getattr(_config, "XGBOOST_OUTPUT_DIR", OUTPUT_DIR / "xgboost")
        )
        CHRONOS_OUTPUT_DIR: Path = Path(
            getattr(_config, "CHRONOS_OUTPUT_DIR", OUTPUT_DIR / "chronos")
        )
        HYBRID_OUTPUT_DIR: Path = Path(
            getattr(_config, "HYBRID_OUTPUT_DIR", OUTPUT_DIR / "hybrid")
        )


P = ParamSpec("P")
R = TypeVar("R")


def setup_directories() -> tuple[Path, Path, Path]:
    """Creates core project directories if they do not already exist.

    Returns:
        Tuple containing `(data_dir, model_dir, output_dir)`.
    """
    data_dir: Path = Path(getattr(Config, "DATA_DIR"))
    model_dir: Path = Path(getattr(Config, "MODEL_DIR"))
    output_dir: Path = Path(getattr(Config, "OUTPUT_DIR"))
    xgboost_output_dir: Path = Path(
        getattr(Config, "XGBOOST_OUTPUT_DIR", output_dir / "xgboost")
    )
    chronos_output_dir: Path = Path(
        getattr(Config, "CHRONOS_OUTPUT_DIR", output_dir / "chronos")
    )
    hybrid_output_dir: Path = Path(
        getattr(Config, "HYBRID_OUTPUT_DIR", output_dir / "hybrid")
    )

    for directory in (
        data_dir,
        model_dir,
        output_dir,
        xgboost_output_dir,
        chronos_output_dir,
        hybrid_output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return data_dir, model_dir, output_dir


def _has_stdout_handler(logger: logging.Logger) -> bool:
    """Checks whether logger already has a stdout stream handler."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            return True
    return False


def _has_file_handler(logger: logging.Logger, file_path: Path) -> bool:
    """Checks whether logger already has a file handler for given path."""
    target: Path = file_path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing: Path = Path(handler.baseFilename).resolve()
            if existing == target:
                return True
    return False


def get_logger(logger_name: str) -> logging.Logger:
    """Builds a deduplicated logger writing to stdout and `training.log`.

    The logger uses a consistent timestamped formatter and INFO level to keep
    operational traces readable while avoiding duplicate log emissions.

    Args:
        logger_name: Logger namespace.

    Returns:
        Configured `logging.Logger` instance.
    """
    _, _, output_dir = setup_directories()

    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _has_stdout_handler(logger):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_file_path: Path = output_dir / "training.log"
    if not _has_file_handler(logger, log_file_path):
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_gpu_memory(logger: logging.Logger) -> None:
    """Logs current CUDA memory allocation/reservation in gigabytes.

    This monitor helps detect when training/inference approaches the 8GB VRAM
    boundary so workloads can be tuned before out-of-memory failures occur.

    Args:
        logger: Active logger instance.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA is not available; GPU memory monitoring is skipped.")
        return

    device_index: int = torch.cuda.current_device()
    device_name: str = torch.cuda.get_device_name(device_index)

    allocated_gb: float = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    reserved_gb: float = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
    total_gb: float = (
        torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    )

    base_message: str = (
        f"GPU [{device_name}] VRAM - "
        f"Allocated: {allocated_gb:.2f} GB | "
        f"Reserved: {reserved_gb:.2f} GB | "
        f"Total: {total_gb:.2f} GB"
    )

    if reserved_gb >= 7.2:
        logger.warning(
            "%s | Warning: VRAM usage is approaching the 8GB hardware limit.",
            base_message,
        )
    else:
        logger.info(base_message)


def _resolve_decorator_logger(
    func: Callable[P, R],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> logging.Logger:
    """Resolves logger target for time decorator in a non-intrusive way."""
    logger_obj: object | None = kwargs.get("logger")
    if isinstance(logger_obj, logging.Logger):
        return logger_obj

    if args:
        bound_obj: object = args[0]
        attr_logger: object = getattr(bound_obj, "logger", None)
        if isinstance(attr_logger, logging.Logger):
            return attr_logger

    return get_logger(func.__module__)


def time_it(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that measures and logs function execution time.

    This enables transparent performance profiling for XGBoost and Chronos
    training/inference workflows without changing business logic.

    Args:
        func: Target callable.

    Returns:
        Wrapped callable preserving original signature.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger: logging.Logger = _resolve_decorator_logger(func, args, kwargs)
        start_time: float = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_seconds: float = time.perf_counter() - start_time
            logger.info("%s executed in %.4f seconds", func.__qualname__, elapsed_seconds)

    return wrapper

