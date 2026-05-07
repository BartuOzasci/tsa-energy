"""Data loading and preprocessing pipeline for multivariate energy forecasting.

The DataLoader class encapsulates ingestion, cleaning, feature engineering,
scaling, chronological splitting, and model-specific formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from src.config import Config
except ImportError:
    from src import config as _config

    class Config:  # type: ignore[too-many-instance-attributes]
        """Compatibility layer when a dedicated Config class is not defined."""

        DATASET_URL: str = str(getattr(_config, "DATASET_URL", ""))
        DATA_DIR: Path = Path(getattr(_config, "DATA_DIR", Path("data")))
        TEST_SIZE: float = float(getattr(_config, "TEST_SIZE", 0.2))
        MAX_LAG_FEATURES: int = int(getattr(_config, "MAX_LAG_FEATURES", 48))
        CONTEXT_LENGTH: int = int(getattr(_config, "CONTEXT_LENGTH", 512))
        PREDICTION_LENGTH: int = int(getattr(_config, "PREDICTION_LENGTH", 24))
        TARGET_COLUMN: str = str(getattr(_config, "TARGET_COLUMN", "target"))
        DATETIME_COLUMN: str = str(getattr(_config, "DATETIME_COLUMN", "timestamp"))


Scaler: TypeAlias = StandardScaler | MinMaxScaler


class DataLoader:
    """Builds a leakage-safe time series data pipeline for XGBoost and Chronos.

    This class keeps all data preparation responsibilities in one cohesive unit.
    """

    def __init__(
        self,
        config: type[Config] = Config,
        target_column: str | None = None,
        datetime_column: str | None = None,
        scaler_kind: Literal["standard", "minmax"] = "standard",
    ) -> None:
        """Initializes pipeline defaults and reusable scaler instances.

        Args:
            config: Configuration source class imported from src.config.
            target_column: Optional target override; defaults to config value.
            datetime_column: Optional datetime override; defaults to config value.
            scaler_kind: Feature/target scaler family.

        Complexity:
            O(1), only assigns references and lightweight objects.
        """
        self.config: type[Config] = config
        self.dataset_url: str = str(getattr(config, "DATASET_URL", ""))
        self.data_dir: Path = Path(getattr(config, "DATA_DIR"))
        self.test_size: float = float(getattr(config, "TEST_SIZE", 0.2))
        self.max_lag_features: int = int(getattr(config, "MAX_LAG_FEATURES", 48))
        self.context_length: int = int(getattr(config, "CONTEXT_LENGTH", 512))
        self.prediction_length: int = int(getattr(config, "PREDICTION_LENGTH", 24))
        self.target_column: str = target_column or str(
            getattr(config, "TARGET_COLUMN", "target")
        )
        self.datetime_column: str = datetime_column or str(
            getattr(config, "DATETIME_COLUMN", "timestamp")
        )

        self.raw_data_path: Path = self.data_dir / "raw_data.csv"
        self.feature_scaler: Scaler = self._build_scaler(scaler_kind)
        self.target_scaler: Scaler = self._build_scaler(scaler_kind)

        self._raw_cache: pd.DataFrame | None = None
        self._cleaned_cache: pd.DataFrame | None = None

    @staticmethod
    def _build_scaler(kind: Literal["standard", "minmax"]) -> Scaler:
        """Creates a scaler instance from a short family name.

        Args:
            kind: Scaler family selector.

        Returns:
            A configured sklearn scaler instance.

        Complexity:
            O(1).
        """
        if kind == "standard":
            return StandardScaler()
        return MinMaxScaler()

    def download_data(self) -> Path:
        """Downloads raw CSV from Config.DATASET_URL with local file caching.

        The file is persisted as DATA_DIR/raw_data.csv and download is skipped
        when cache already exists.

        Returns:
            Absolute path to cached raw CSV file.

        Complexity:
            O(n) due to CSV network read and disk write.
        """
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        if self.raw_data_path.exists():
            return self.raw_data_path

        if not self.dataset_url:
            raise ValueError("Config.DATASET_URL is empty. Please define a valid source URL.")

        raw_df: pd.DataFrame = pd.read_csv(self.dataset_url)
        raw_df.to_csv(self.raw_data_path, index=False)
        return self.raw_data_path

    def load_raw_data(self) -> pd.DataFrame:
        """Loads cached raw data into a DataFrame.

        Returns:
            Raw dataset as pandas DataFrame.

        Complexity:
            O(n) with n as row count.
        """
        if self._raw_cache is None:
            raw_path: Path = self.download_data()
            self._raw_cache = pd.read_csv(raw_path)
        return self._raw_cache.copy()

    def _get_cleaned(self) -> pd.DataFrame:
        """Returns preprocessed DataFrame, computing and caching on first call."""
        if self._cleaned_cache is None:
            self._cleaned_cache = self.preprocess_data(self.load_raw_data())
        return self._cleaned_cache

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans a raw dataset and converts it into indexed time series format.

        Steps:
            1) Parse datetime column.
            2) Sort and set datetime index.
            3) Convert values to numeric when possible.
            4) Repair gaps with interpolation, then ffill/bfill fallback.

        Args:
            data: Raw input DataFrame.

        Returns:
            Cleaned DataFrame indexed by datetime.

        Complexity:
            O(n * m), where n is row count and m is column count.
        """
        frame: pd.DataFrame = data.copy()

        if self.datetime_column not in frame.columns:
            raise KeyError(f"Datetime column '{self.datetime_column}' not found in data.")
        if self.target_column not in frame.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in data.")

        frame[self.datetime_column] = pd.to_datetime(
            frame[self.datetime_column],
            errors="coerce",
        )
        frame = frame.dropna(subset=[self.datetime_column])
        frame = frame.sort_values(self.datetime_column)
        frame = frame.set_index(self.datetime_column)
        frame = frame[~frame.index.duplicated(keep="last")]

        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.interpolate(method="time", limit_direction="both")
        frame = frame.ffill().bfill()
        frame = frame.dropna(how="any")

        return frame

    def _lag_steps(self) -> list[int]:
        """Builds lag step values including dense recent lags and seasonal anchors.

        Dense range covers max_lag_features steps; seasonal anchors (daily for
        both hourly/30-min data, and weekly for both resolutions) are always
        included regardless of max_lag to capture periodic structure.

        Returns:
            Sorted lag step list.

        Complexity:
            O(L), where L is max lag count.
        """
        max_lag: int = max(1, self.max_lag_features)
        lag_steps: set[int] = set(range(1, max_lag + 1))
        # Seasonal anchors: daily (24h/48×30min) and weekly (168h/336×30min)
        lag_steps.update({1, 2, 24, 48, 168, 336})
        return sorted(lag_steps)

    @staticmethod
    def _rolling_windows(max_lag: int) -> list[int]:
        """Selects rolling window sizes bounded by lag capacity.

        Args:
            max_lag: Maximum lag size from configuration.

        Returns:
            Rolling window lengths suitable for statistical features.

        Complexity:
            O(1), fixed candidate list.
        """
        candidates: tuple[int, ...] = (3, 6, 12, 24, 48)
        windows: list[int] = [window for window in candidates if window <= max_lag]
        return windows if windows else [3]

    def _add_time_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Creates calendar-based features from DatetimeIndex.

        Args:
            frame: Datetime-indexed DataFrame.

        Returns:
            DataFrame enriched with temporal columns.

        Complexity:
            O(n).
        """
        idx: pd.DatetimeIndex = frame.index  # type: ignore[assignment]
        frame["hour"] = idx.hour.astype(np.int16)
        frame["day_of_week"] = idx.dayofweek.astype(np.int16)
        frame["month"] = idx.month.astype(np.int16)
        frame["quarter"] = idx.quarter.astype(np.int16)
        frame["day_of_year"] = idx.dayofyear.astype(np.int16)
        frame["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)
        return frame

    def _add_lag_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds leakage-safe target lag columns for tabular forecasting.

        Args:
            frame: Preprocessed DataFrame.

        Returns:
            DataFrame with lag feature columns.

        Complexity:
            O(n * L), where L is number of lag features.
        """
        for lag in self._lag_steps():
            frame[f"{self.target_column}_lag_{lag}"] = frame[self.target_column].shift(lag)
        return frame

    def _add_rolling_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds rolling mean/std features using shifted history to avoid leakage.

        Args:
            frame: DataFrame with target column.

        Returns:
            DataFrame with rolling statistical feature columns.

        Complexity:
            O(n * W), where W is number of rolling windows.
        """
        shifted_target: pd.Series = frame[self.target_column].shift(1)
        for window in self._rolling_windows(self.max_lag_features):
            frame[f"{self.target_column}_roll_mean_{window}"] = (
                shifted_target.rolling(window=window, min_periods=window).mean()
            )
            frame[f"{self.target_column}_roll_std_{window}"] = (
                shifted_target.rolling(window=window, min_periods=window).std()
            )
        return frame

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Builds temporal, lag, and rolling features for XGBoost-style models.

        Args:
            data: Cleaned and datetime-indexed data.

        Returns:
            Feature-engineered DataFrame with valid rows only.

        Complexity:
            O(n * (L + W)).
        """
        frame: pd.DataFrame = data.copy()
        frame = self._add_time_features(frame)
        frame = self._add_lag_features(frame)
        frame = self._add_rolling_features(frame)
        frame = frame.dropna(how="any")
        return frame

    def split_train_test(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data chronologically using Config.TEST_SIZE without shuffling.

        Args:
            data: Time-indexed DataFrame.

        Returns:
            train_df, test_df in chronological order.

        Complexity:
            O(n), slice-based split.
        """
        if data.empty:
            raise ValueError("Cannot split an empty dataset.")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("Config.TEST_SIZE must be in the (0, 1) interval.")

        split_idx: int = int(len(data) * (1.0 - self.test_size))
        split_idx = min(max(split_idx, 1), len(data) - 1)

        train_df: pd.DataFrame = data.iloc[:split_idx].copy()
        test_df: pd.DataFrame = data.iloc[split_idx:].copy()
        return train_df, test_df

    def _split_features_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separates predictor matrix and target vector.

        Args:
            data: Feature-engineered DataFrame.

        Returns:
            X, y pair.

        Complexity:
            O(m), where m is number of columns.
        """
        if self.target_column not in data.columns:
            raise KeyError(f"Target column '{self.target_column}' not found after preprocessing.")

        x_frame: pd.DataFrame = data.drop(columns=[self.target_column])
        y_series: pd.Series = data[self.target_column]
        return x_frame, y_series

    def inverse_transform_target(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        """Converts scaled target predictions back to original value space.

        Args:
            values: Scaled target predictions.

        Returns:
            Inverse-transformed 1D NumPy array.

        Complexity:
            O(n).
        """
        if not hasattr(self.target_scaler, "scale_"):
            raise RuntimeError("Target scaler is not fitted. Call get_xgboost_data first.")

        array: np.ndarray
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)

        reshaped: np.ndarray = array.reshape(-1, 1)
        restored: np.ndarray = self.target_scaler.inverse_transform(reshaped)
        return restored.reshape(-1)

    def get_xgboost_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Produces leakage-safe tabular train/test arrays for XGBoost.

        Workflow:
            ingestion -> preprocessing -> feature engineering -> chronological split
            -> train-only scaler fitting -> transform train/test.

        Returns:
            X_train, y_train, X_test, y_test as float32 NumPy arrays.

        Complexity:
            O(n * (L + W + m)) for feature creation and scaling.
        """
        cleaned_df: pd.DataFrame = self._get_cleaned()
        featured_df: pd.DataFrame = self.engineer_features(cleaned_df)

        train_df, test_df = self.split_train_test(featured_df)
        x_train_df, y_train_series = self._split_features_target(train_df)
        x_test_df, y_test_series = self._split_features_target(test_df)

        self.feature_scaler.fit(x_train_df)
        self.target_scaler.fit(y_train_series.to_numpy().reshape(-1, 1))

        x_train: np.ndarray = self.feature_scaler.transform(x_train_df).astype(np.float32)
        x_test: np.ndarray = self.feature_scaler.transform(x_test_df).astype(np.float32)
        y_train: np.ndarray = (
            self.target_scaler.transform(y_train_series.to_numpy().reshape(-1, 1))
            .reshape(-1)
            .astype(np.float32)
        )
        y_test: np.ndarray = (
            self.target_scaler.transform(y_test_series.to_numpy().reshape(-1, 1))
            .reshape(-1)
            .astype(np.float32)
        )

        return x_train, y_train, x_test, y_test

    def _build_chronos_windows(
        self,
        series: np.ndarray,
        stride: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Converts a 1D time series into context/horizon window pairs.

        Args:
            series: One-dimensional target array.
            stride: Step between consecutive windows. Defaults to
                prediction_length (non-overlapping) for efficient batched
                inference. Use stride=1 for dense sliding-window evaluation.

        Returns:
            contexts, horizons arrays aligned for Chronos-style forecasting.

        Complexity:
            O(n/stride * (context_length + prediction_length)).
        """
        effective_stride: int = self.prediction_length if stride is None else max(1, stride)
        min_required: int = self.context_length + self.prediction_length
        if series.size < min_required:
            raise ValueError(
                "Series is too short for Chronos windows: "
                f"need at least {min_required}, got {series.size}."
            )

        contexts: list[np.ndarray] = []
        horizons: list[np.ndarray] = []
        for end_idx in range(
            self.context_length,
            series.size - self.prediction_length + 1,
            effective_stride,
        ):
            start_idx: int = end_idx - self.context_length
            contexts.append(series[start_idx:end_idx])
            horizons.append(series[end_idx : end_idx + self.prediction_length])

        return np.asarray(contexts, dtype=np.float32), np.asarray(horizons, dtype=np.float32)

    def get_chronos_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares Chronos-compatible context/horizon tensors.

        The method keeps chronological integrity and uses trailing training
        history as context for the first test windows.

        Returns:
            train_contexts, train_horizons, test_contexts, test_horizons.

        Complexity:
            O(n * (context_length + prediction_length)).
        """
        cleaned_df: pd.DataFrame = self._get_cleaned()

        series: np.ndarray = cleaned_df[self.target_column].to_numpy(dtype=np.float32)
        split_idx: int = int(series.size * (1.0 - self.test_size))
        split_idx = min(max(split_idx, self.context_length + self.prediction_length), series.size - 1)

        train_series: np.ndarray = series[:split_idx]
        test_source_start: int = max(0, split_idx - self.context_length)
        test_series_source: np.ndarray = series[test_source_start:]

        train_contexts_np, train_horizons_np = self._build_chronos_windows(train_series)
        test_contexts_np, test_horizons_np = self._build_chronos_windows(test_series_source)

        train_contexts: torch.Tensor = torch.tensor(train_contexts_np, dtype=torch.float32)
        train_horizons: torch.Tensor = torch.tensor(train_horizons_np, dtype=torch.float32)
        test_contexts: torch.Tensor = torch.tensor(test_contexts_np, dtype=torch.float32)
        test_horizons: torch.Tensor = torch.tensor(test_horizons_np, dtype=torch.float32)

        return train_contexts, train_horizons, test_contexts, test_horizons

