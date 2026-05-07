"""Centralized configuration for the multivariate energy TSA project.

This module keeps paths, hardware runtime options, model hyperparameters,
and reproducibility settings in one place.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Defines project directory layout with explicit subfolders."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    xgboost_model_dir: Path = field(init=False)
    chronos_model_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    xgboost_output_dir: Path = field(init=False)
    chronos_output_dir: Path = field(init=False)
    hybrid_output_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        root: Path = self.base_dir
        object.__setattr__(self, "data_dir", root / "data")
        object.__setattr__(self, "raw_data_dir", root / "data" / "raw")
        object.__setattr__(self, "processed_data_dir", root / "data" / "processed")
        object.__setattr__(self, "model_dir", root / "models")
        object.__setattr__(self, "xgboost_model_dir", root / "models" / "xgboost")
        object.__setattr__(self, "chronos_model_dir", root / "models" / "chronos")
        object.__setattr__(self, "output_dir", root / "outputs")
        object.__setattr__(self, "xgboost_output_dir", root / "outputs" / "xgboost")
        object.__setattr__(self, "chronos_output_dir", root / "outputs" / "chronos")
        object.__setattr__(self, "hybrid_output_dir", root / "outputs" / "hybrid")
        object.__setattr__(self, "metrics_dir", root / "outputs" / "metrics")
        object.__setattr__(self, "plots_dir", root / "outputs" / "plots")
        object.__setattr__(self, "log_dir", root / "logs")

    def all_directories(self) -> tuple[Path, ...]:
        """Returns all directories that must exist before pipeline execution."""
        return (
            self.base_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.model_dir,
            self.xgboost_model_dir,
            self.chronos_model_dir,
            self.output_dir,
            self.xgboost_output_dir,
            self.chronos_output_dir,
            self.hybrid_output_dir,
            self.metrics_dir,
            self.plots_dir,
            self.log_dir,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Applies CUDA-aware runtime settings for stable 8GB VRAM execution."""

    prefer_cuda: bool = True

    @property
    def device(self) -> torch.device:
        """Selects CUDA when available, otherwise falls back to CPU."""
        use_cuda: bool = self.prefer_cuda and torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")

    def apply_torch_optimizations(self) -> None:
        """Enables memory and throughput tuning knobs for PyTorch."""
        if self.device.type != "cuda":
            return

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


@dataclass(frozen=True, slots=True)
class XGBoostConfig:
    """GPU-accelerated XGBoost defaults for multivariate regression."""

    tree_method: str = "hist"
    device: str = "cuda"
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    learning_rate: float = 0.025
    max_depth: int = 8
    n_estimators: int = 1800
    subsample: float = 0.90
    colsample_bytree: float = 0.85
    min_child_weight: float = 3.0
    gamma: float = 0.05
    reg_alpha: float = 0.02
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 80
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Exports a plain dict to plug directly into training code."""
        return {
            "tree_method": self.tree_method,
            "device": self.device,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            # XGBoost consumes seed values from estimator parameters.
            "seed": self.seed,
            "random_state": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ChronosConfig:
    """Chronos-T5 defaults tuned for 8GB VRAM inference/training stability."""

    model_id: str = "amazon/chronos-t5-base"
    torch_dtype: torch.dtype = torch.bfloat16
    context_length: int = 1024
    prediction_length: int = 24
    batch_size: int = 16
    num_samples: int = 32
    confidence_level: float = 0.9


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Data split and feature-engineering controls for TSA pipelines."""

    test_size: float = 0.2
    sliding_window_size: int = 512
    max_lag_features: int = 48
    hybrid_xgb_weight: float = 0.37
    hybrid_auto_weight_tuning: bool = True
    hybrid_weight_search_points: int = 101
    seed: int = 42


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Root configuration object that wires all modular config sections."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    chronos: ChronosConfig = field(default_factory=ChronosConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def initialize(self) -> None:
        """Creates required directories and applies deterministic runtime setup."""
        self.init_directories()
        self.runtime.apply_torch_optimizations()
        seed_everything(self.pipeline.seed)

    def init_directories(self) -> None:
        """Ensures all required project directories exist."""
        for directory in self.paths.all_directories():
            directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch; XGBoost is seeded via params."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


CONFIG: Final[AppConfig] = AppConfig()
CONFIG.initialize()

# Path exports
BASE_DIR: Final[Path] = CONFIG.paths.base_dir
DATA_DIR: Final[Path] = CONFIG.paths.data_dir
RAW_DATA_DIR: Final[Path] = CONFIG.paths.raw_data_dir
PROCESSED_DATA_DIR: Final[Path] = CONFIG.paths.processed_data_dir
MODEL_DIR: Final[Path] = CONFIG.paths.model_dir
XGBOOST_MODEL_DIR: Final[Path] = CONFIG.paths.xgboost_model_dir
CHRONOS_MODEL_DIR: Final[Path] = CONFIG.paths.chronos_model_dir
OUTPUT_DIR: Final[Path] = CONFIG.paths.output_dir
XGBOOST_OUTPUT_DIR: Final[Path] = CONFIG.paths.xgboost_output_dir
CHRONOS_OUTPUT_DIR: Final[Path] = CONFIG.paths.chronos_output_dir
HYBRID_OUTPUT_DIR: Final[Path] = CONFIG.paths.hybrid_output_dir
METRICS_DIR: Final[Path] = CONFIG.paths.metrics_dir
PLOTS_DIR: Final[Path] = CONFIG.paths.plots_dir
LOG_DIR: Final[Path] = CONFIG.paths.log_dir

# Runtime export
DEVICE: Final[torch.device] = CONFIG.runtime.device

# Pipeline exports
SEED: Final[int] = CONFIG.pipeline.seed
TEST_SIZE: Final[float] = CONFIG.pipeline.test_size
SLIDING_WINDOW_SIZE: Final[int] = CONFIG.pipeline.sliding_window_size
MAX_LAG_FEATURES: Final[int] = CONFIG.pipeline.max_lag_features
HYBRID_XGB_WEIGHT: Final[float] = CONFIG.pipeline.hybrid_xgb_weight
HYBRID_AUTO_WEIGHT_TUNING: Final[bool] = CONFIG.pipeline.hybrid_auto_weight_tuning
HYBRID_WEIGHT_SEARCH_POINTS: Final[int] = CONFIG.pipeline.hybrid_weight_search_points

# Data source and schema exports
DATASET_URL: Final[str] = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
)
DATETIME_COLUMN: Final[str] = "timestamp"
TARGET_COLUMN: Final[str] = "value"

# Chronos exports
CHRONOS_MODEL_ID: Final[str] = CONFIG.chronos.model_id
CHRONOS_TORCH_DTYPE: Final[torch.dtype] = CONFIG.chronos.torch_dtype
CONTEXT_LENGTH: Final[int] = CONFIG.chronos.context_length
PREDICTION_LENGTH: Final[int] = CONFIG.chronos.prediction_length
CHRONOS_BATCH_SIZE: Final[int] = CONFIG.chronos.batch_size
CHRONOS_NUM_SAMPLES: Final[int] = CONFIG.chronos.num_samples
CHRONOS_CONFIDENCE_LEVEL: Final[float] = CONFIG.chronos.confidence_level

# XGBoost exports
XGBOOST_PARAMS: Final[dict[str, Any]] = CONFIG.xgboost.to_dict()
XGB_EARLY_STOPPING_ROUNDS: Final[int] = CONFIG.xgboost.early_stopping_rounds

# Backward-compatible aliases
MODELS_DIR: Final[Path] = MODEL_DIR
OUTPUTS_DIR: Final[Path] = OUTPUT_DIR
XGB_PARAMS: Final[dict[str, Any]] = XGBOOST_PARAMS


class Config:
    """Class-based compatibility facade for project settings.

    Several modules import settings via `from src.config import Config`.
    This class mirrors module-level constants to keep a single source of truth.
    """

    BASE_DIR: Final[Path] = BASE_DIR
    DATA_DIR: Final[Path] = DATA_DIR
    RAW_DATA_DIR: Final[Path] = RAW_DATA_DIR
    PROCESSED_DATA_DIR: Final[Path] = PROCESSED_DATA_DIR
    MODEL_DIR: Final[Path] = MODEL_DIR
    XGBOOST_MODEL_DIR: Final[Path] = XGBOOST_MODEL_DIR
    CHRONOS_MODEL_DIR: Final[Path] = CHRONOS_MODEL_DIR
    OUTPUT_DIR: Final[Path] = OUTPUT_DIR
    XGBOOST_OUTPUT_DIR: Final[Path] = XGBOOST_OUTPUT_DIR
    CHRONOS_OUTPUT_DIR: Final[Path] = CHRONOS_OUTPUT_DIR
    HYBRID_OUTPUT_DIR: Final[Path] = HYBRID_OUTPUT_DIR
    METRICS_DIR: Final[Path] = METRICS_DIR
    PLOTS_DIR: Final[Path] = PLOTS_DIR
    LOG_DIR: Final[Path] = LOG_DIR

    DEVICE: Final[torch.device] = DEVICE

    SEED: Final[int] = SEED
    TEST_SIZE: Final[float] = TEST_SIZE
    SLIDING_WINDOW_SIZE: Final[int] = SLIDING_WINDOW_SIZE
    MAX_LAG_FEATURES: Final[int] = MAX_LAG_FEATURES
    HYBRID_XGB_WEIGHT: Final[float] = HYBRID_XGB_WEIGHT
    HYBRID_AUTO_WEIGHT_TUNING: Final[bool] = HYBRID_AUTO_WEIGHT_TUNING
    HYBRID_WEIGHT_SEARCH_POINTS: Final[int] = HYBRID_WEIGHT_SEARCH_POINTS

    DATASET_URL: Final[str] = DATASET_URL
    DATETIME_COLUMN: Final[str] = DATETIME_COLUMN
    TARGET_COLUMN: Final[str] = TARGET_COLUMN

    CHRONOS_MODEL_ID: Final[str] = CHRONOS_MODEL_ID
    CHRONOS_TORCH_DTYPE: Final[torch.dtype] = CHRONOS_TORCH_DTYPE
    CONTEXT_LENGTH: Final[int] = CONTEXT_LENGTH
    PREDICTION_LENGTH: Final[int] = PREDICTION_LENGTH
    CHRONOS_BATCH_SIZE: Final[int] = CHRONOS_BATCH_SIZE
    CHRONOS_NUM_SAMPLES: Final[int] = CHRONOS_NUM_SAMPLES
    CHRONOS_CONFIDENCE_LEVEL: Final[float] = CHRONOS_CONFIDENCE_LEVEL

    XGB_PARAMS: Final[dict[str, Any]] = XGB_PARAMS
    XGB_EARLY_STOPPING_ROUNDS: Final[int] = XGB_EARLY_STOPPING_ROUNDS
