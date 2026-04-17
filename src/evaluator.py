"""Evaluation utilities for point and probabilistic time series forecasts.

This module standardizes metric computation and visualization for comparing
deterministic models (e.g., XGBoost) against probabilistic foundation models
(e.g., Chronos).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # pyright: ignore[reportMissingImports]  # type: ignore[import-not-found]
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

try:
    from src.config import Config
except ImportError:
    from src import config as _config

    class Config:  # type: ignore[too-many-instance-attributes]
        """Compatibility layer when src.config exposes module-level constants."""

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


class ModelEvaluator:
    """Evaluates and visualizes model outputs for TSA model governance.

    RMSE emphasizes larger errors, MAE captures typical absolute deviation, and
    MAPE offers scale-free relative error insight. Together they provide a
    balanced view for energy-load forecasting quality assessment.
    """

    def __init__(self, config: type[Config] = Config) -> None:
        """Initializes evaluator state and plotting style.

        Args:
            config: Configuration source with output directory information.
        """
        self.config: type[Config] = config
        self.output_dir: Path = Path(getattr(config, "OUTPUT_DIR"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dirs: dict[str, Path] = {
            "xgboost": Path(
                getattr(config, "XGBOOST_OUTPUT_DIR", self.output_dir / "xgboost")
            ),
            "chronos": Path(
                getattr(config, "CHRONOS_OUTPUT_DIR", self.output_dir / "chronos")
            ),
            "hybrid": Path(getattr(config, "HYBRID_OUTPUT_DIR", self.output_dir / "hybrid")),
        }
        for directory in self.model_output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        # Use an academic, publication-friendly style across all generated plots.
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            sns.set_theme(style="whitegrid")
        else:
            sns.set_theme(style="whitegrid")

    def _resolve_model_output_dir(self, model_name: str) -> Path:
        """Resolves and creates the output folder for a model namespace."""
        key: str = model_name.strip().lower()
        model_dir: Path = self.model_output_dirs.get(key, self.output_dir / key)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def _to_numpy_1d(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
        """Converts input values to a flattened NumPy array.

        Args:
            values: Input numeric sequence.

        Returns:
            One-dimensional NumPy array.
        """
        array: np.ndarray = np.asarray(values, dtype=np.float64)
        return array.reshape(-1)

    @staticmethod
    def _validate_equal_length(*arrays: np.ndarray) -> None:
        """Validates that all arrays have identical length.

        Args:
            arrays: Arrays to compare.
        """
        if not arrays:
            raise ValueError("At least one array is required for length validation.")

        expected_length: int = arrays[0].shape[0]
        for array in arrays[1:]:
            if array.shape[0] != expected_length:
                raise ValueError("All input arrays must have the same length.")

    @staticmethod
    def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes MAPE (%) with zero-safe denominator handling.

        Zero targets are replaced with machine epsilon to avoid division issues
        while preserving compatibility with sklearn's MAPE implementation.
        """
        epsilon: float = float(np.finfo(np.float64).eps)
        safe_true: np.ndarray = np.where(np.isclose(y_true, 0.0), epsilon, y_true)
        mape_ratio: float = float(mean_absolute_percentage_error(safe_true, y_pred))
        return mape_ratio * 100.0

    def _serialize_metrics(self, metrics: Mapping[str, float], model_name: str) -> Path:
        """Persists computed metrics as JSON in the output directory.

        Args:
            metrics: Metric dictionary to serialize.
            model_name: Model identifier used for the output filename.

        Returns:
            Written metrics file path.
        """
        model_output_dir: Path = self._resolve_model_output_dir(model_name)
        metrics_path: Path = model_output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(dict(metrics), file, indent=2)
        return metrics_path

    def calculate_metrics(
        self,
        y_true: Sequence[float] | np.ndarray | pd.Series,
        y_pred: Sequence[float] | np.ndarray | pd.Series,
        model_name: str,
    ) -> dict[str, float]:
        """Calculates core regression metrics and stores them as JSON.

        Metrics are selected to jointly capture average error magnitude,
        sensitivity to large deviations, and relative performance in percentage
        terms for cross-model comparability.

        Args:
            y_true: Ground-truth values.
            y_pred: Predicted values.
            model_name: Name used for output file naming.

        Returns:
            Dictionary with `mse`, `rmse`, `mae`, `mape`, and `r2`.
        """
        y_true_arr: np.ndarray = self._to_numpy_1d(y_true)
        y_pred_arr: np.ndarray = self._to_numpy_1d(y_pred)
        self._validate_equal_length(y_true_arr, y_pred_arr)

        mse: float = float(mean_squared_error(y_true_arr, y_pred_arr))
        rmse: float = float(np.sqrt(mse))
        mae: float = float(mean_absolute_error(y_true_arr, y_pred_arr))
        mape: float = self._safe_mape(y_true_arr, y_pred_arr)
        r2: float = float(r2_score(y_true_arr, y_pred_arr))

        metrics: dict[str, float] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }
        self._serialize_metrics(metrics=metrics, model_name=model_name)
        return metrics

    def plot_predictions(
        self,
        y_true: Sequence[float] | np.ndarray | pd.Series,
        y_pred: Sequence[float] | np.ndarray | pd.Series,
        timestamps: Sequence[object] | np.ndarray | pd.Index | pd.Series,
        model_name: str,
    ) -> Path:
        """Plots point forecasts against actual observations.

        Args:
            y_true: Ground-truth values.
            y_pred: Point forecast values.
            timestamps: Time axis values aligned with targets.
            model_name: Model identifier for output file naming.

        Returns:
            Saved figure file path.
        """
        y_true_arr: np.ndarray = self._to_numpy_1d(y_true)
        y_pred_arr: np.ndarray = self._to_numpy_1d(y_pred)
        time_arr: np.ndarray = np.asarray(timestamps)

        self._validate_equal_length(y_true_arr, y_pred_arr)
        if time_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError("timestamps length must match y_true and y_pred length.")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(time_arr, y_true_arr, label="Actual", linewidth=2.0, linestyle="-")
        ax.plot(time_arr, y_pred_arr, label="Predicted", linewidth=2.0, linestyle="--")
        ax.set_title(f"{model_name} Forecast vs Actual", fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy Load")
        ax.legend(loc="best")
        fig.autofmt_xdate()

        model_output_dir: Path = self._resolve_model_output_dir(model_name)
        plot_path: Path = model_output_dir / "forecast.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def plot_probabilistic_predictions(
        self,
        y_true: Sequence[float] | np.ndarray | pd.Series,
        median_pred: Sequence[float] | np.ndarray | pd.Series,
        low_band: Sequence[float] | np.ndarray | pd.Series,
        high_band: Sequence[float] | np.ndarray | pd.Series,
        timestamps: Sequence[object] | np.ndarray | pd.Index | pd.Series,
        model_name: str = "chronos",
    ) -> Path:
        """Plots median trajectory and confidence interval bands.

        Confidence bands communicate forecast uncertainty, which is essential for
        risk-aware decision-making in energy planning and operations.

        Args:
            y_true: Ground-truth values.
            median_pred: Median forecast trajectory.
            low_band: Lower confidence bound.
            high_band: Upper confidence bound.
            timestamps: Time axis values.
            model_name: Model identifier for output file naming.

        Returns:
            Saved figure file path.
        """
        y_true_arr: np.ndarray = self._to_numpy_1d(y_true)
        median_arr: np.ndarray = self._to_numpy_1d(median_pred)
        low_arr: np.ndarray = self._to_numpy_1d(low_band)
        high_arr: np.ndarray = self._to_numpy_1d(high_band)
        time_arr: np.ndarray = np.asarray(timestamps)

        self._validate_equal_length(y_true_arr, median_arr, low_arr, high_arr)
        if time_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                "timestamps length must match y_true and probabilistic prediction arrays."
            )

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(time_arr, y_true_arr, color="black", linewidth=2.1, label="Actual")
        ax.plot(
            time_arr,
            median_arr,
            color="#1f77b4",
            linewidth=2.0,
            label="Median Forecast",
        )
        ax.fill_between(
            time_arr,
            low_arr,
            high_arr,
            color="#1f77b4",
            alpha=0.22,
            label="Confidence Interval",
        )

        ax.set_title(f"{model_name} Probabilistic Forecast", fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy Load")
        ax.legend(loc="best")
        fig.autofmt_xdate()

        model_output_dir: Path = self._resolve_model_output_dir(model_name)
        plot_path: Path = model_output_dir / "probabilistic_forecast.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def compare_models(self, metrics_dict: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
        """Builds a side-by-side model comparison report and exports CSV.

        Args:
            metrics_dict: Mapping such as
                {"xgboost": {"rmse": ...}, "chronos": {"rmse": ...}}.

        Returns:
            Comparison DataFrame indexed by model names.
        """
        if not metrics_dict:
            raise ValueError("metrics_dict cannot be empty.")

        comparison_df: pd.DataFrame = pd.DataFrame.from_dict(metrics_dict, orient="index")

        preferred_order: list[str] = ["mse", "rmse", "mae", "mape", "r2"]
        ordered_columns: list[str] = [
            col for col in preferred_order if col in comparison_df.columns
        ] + [col for col in comparison_df.columns if col not in preferred_order]
        comparison_df = comparison_df[ordered_columns]

        print(comparison_df.to_string())

        csv_path: Path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=True)
        return comparison_df

