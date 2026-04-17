"""XGBoost model orchestration for multivariate energy time series forecasting.

This module centralizes model lifecycle operations such as initialization,
training, inference, evaluation, persistence, and feature-importance analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb  # pyright: ignore[reportMissingImports]  # type: ignore[import-not-found]
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:

    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Backward-compatible RMSE helper for older sklearn versions."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

try:
    from src.config import Config
except ImportError:
    from src import config as _config

    class Config:  # type: ignore[too-many-instance-attributes]
        """Compatibility layer when src.config exposes module-level constants."""

        XGB_PARAMS: dict[str, Any] = dict(
            getattr(_config, "XGB_PARAMS", getattr(_config, "XGBOOST_PARAMS", {}))
        )
        MODEL_DIR: Path = Path(getattr(_config, "MODEL_DIR", Path("models")))


class XGBoostModel:
    """Encapsulates GPU-optimized XGBoost training and model management.

    The class ensures that CUDA-focused defaults and early stopping policies are
    consistently applied to prevent unnecessary VRAM usage and overfitting.
    """

    def __init__(
        self,
        config: type[Config] = Config,
        early_stopping_rounds: int = 50,
    ) -> None:
        """Initializes an XGBRegressor using centralized configuration.

        GPU-focused defaults are enforced because `tree_method='hist'` and
        `device='cuda'` offer the most stable speed/memory trade-off on modern
        NVIDIA cards with limited VRAM.

        Args:
            config: Config class provider for params and model directory.
            early_stopping_rounds: Patience for validation-based stopping.
        """
        self.config: type[Config] = config
        self.early_stopping_rounds: int = early_stopping_rounds
        self.model_dir: Path = Path(getattr(config, "MODEL_DIR"))

        params: dict[str, Any] = dict(getattr(config, "XGB_PARAMS", {}))
        if not params:
            raise ValueError("Config.XGB_PARAMS is empty. Provide valid XGBoost params.")

        # Enforce CUDA-oriented settings expected by the project architecture.
        params["tree_method"] = "hist"
        params["device"] = "cuda"

        self.params: dict[str, Any] = params
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(**self.params)
        self.is_fitted: bool = False

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> xgb.XGBRegressor:
        """Fits the model using validation-driven early stopping.

        Early stopping is critical for TSA workloads because it prevents late
        boosting rounds from overfitting to temporal noise while reducing GPU
        compute time and memory pressure.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix.
            y_val: Validation target vector.

        Returns:
            The fitted XGBRegressor instance.
        """
        self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.is_fitted = True
        return self.model

    def predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generates predictions as a NumPy array for downstream compatibility.

        Args:
            X_test: Test feature matrix.

        Returns:
            Predicted values in one-dimensional NumPy format.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        try:
            booster: xgb.Booster = self.model.get_booster()
            dtest: xgb.DMatrix = xgb.DMatrix(X_test)
            best_iteration: int | None = getattr(self.model, "best_iteration", None)
            if best_iteration is not None and best_iteration >= 0:
                raw_predictions = booster.predict(
                    dtest,
                    iteration_range=(0, best_iteration + 1),
                )
            else:
                raw_predictions = booster.predict(dtest)
        except Exception:
            raw_predictions = self.model.predict(X_test)

        predictions: np.ndarray = np.asarray(raw_predictions, dtype=np.float64)
        return predictions.reshape(-1)

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Computes regression quality metrics for model monitoring.

        Args:
            y_true: Ground-truth targets.
            y_pred: Predicted targets.

        Returns:
            Dictionary containing MSE, RMSE, and MAE scores.
        """
        y_true_array: np.ndarray = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred_array: np.ndarray = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        mse: float = float(mean_squared_error(y_true_array, y_pred_array))
        rmse: float = float(root_mean_squared_error(y_true_array, y_pred_array))
        mae: float = float(mean_absolute_error(y_true_array, y_pred_array))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

    def save_model(self, model_name: str = "xgboost_tsa_model.json") -> Path:
        """Persists the trained model in JSON format for safe portability.

        JSON is preferred over legacy binary serialization due to improved
        forward-compatibility across XGBoost versions and safer model exchange.

        Args:
            model_name: Output model filename.

        Returns:
            Absolute saved model path.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Cannot save an unfitted model.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path: Path = self.model_dir / model_name
        self.model.save_model(model_path)
        return model_path

    def load_model(self, model_path: str | Path) -> xgb.XGBRegressor:
        """Loads a saved JSON model back into a CUDA-configured regressor.

        Args:
            model_path: Path to serialized model file.

        Returns:
            Reconstructed XGBRegressor ready for inference.
        """
        resolved_path: Path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {resolved_path}")

        reloaded_model: xgb.XGBRegressor = xgb.XGBRegressor(**self.params)
        reloaded_model.load_model(resolved_path)

        self.model = reloaded_model
        self.is_fitted = True
        return self.model

    def get_feature_importance(self, feature_names: Sequence[str]) -> pd.DataFrame:
        """Returns sorted feature importance table for lag and temporal analysis.

        XGBoost reports importances by internal names (`f0`, `f1`, ...). This
        method maps those indices back to original feature names and returns both
        `weight` and `gain` signals to support robust interpretability.

        Args:
            feature_names: Original training feature names in column order.

        Returns:
            DataFrame sorted by gain (then weight) in descending order.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not trained. Train or load a model before importance analysis."
            )

        booster: xgb.Booster = self.model.get_booster()
        weight_scores: Mapping[str, float] = booster.get_score(importance_type="weight")
        gain_scores: Mapping[str, float] = booster.get_score(importance_type="gain")

        records: list[dict[str, float | str]] = []
        for index, feature in enumerate(feature_names):
            key: str = f"f{index}"
            records.append(
                {
                    "feature": feature,
                    "weight": float(weight_scores.get(key, 0.0)),
                    "gain": float(gain_scores.get(key, 0.0)),
                }
            )

        importance_df: pd.DataFrame = pd.DataFrame(records)
        importance_df = importance_df.sort_values(
            by=["gain", "weight"],
            ascending=False,
        ).reset_index(drop=True)
        return importance_df

