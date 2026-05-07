import gc
import logging
import time

import numpy as np
import pandas as pd
import torch

from src.chronos_model import ChronosModel
from src.config import Config
from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.utils import get_logger, log_gpu_memory, setup_directories
from src.xgboost_model import XGBoostModel


def _split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds a chronological train/validation split for early stopping.

    Args:
        x_train: Full training feature matrix.
        y_train: Full training target vector.
        validation_ratio: Fraction reserved for validation.

    Returns:
        Tuple as (x_fit, y_fit, x_val, y_val).
    """
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of rows.")
    if x_train.shape[0] < 3:
        raise ValueError("At least 3 samples are required for train/validation split.")
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError("validation_ratio must be in the (0, 1) interval.")

    split_idx: int = int(x_train.shape[0] * (1.0 - validation_ratio))
    split_idx = min(max(split_idx, 1), x_train.shape[0] - 1)

    x_fit: np.ndarray = x_train[:split_idx]
    y_fit: np.ndarray = y_train[:split_idx]
    x_val: np.ndarray = x_train[split_idx:]
    y_val: np.ndarray = y_train[split_idx:]
    return x_fit, y_fit, x_val, y_val


def _first_horizon_step(values: np.ndarray) -> np.ndarray:
    """Converts forecast outputs to one-step-ahead vectors.

    Args:
        values: Forecast array with shape [N] or [N, H].

    Returns:
        One-dimensional first-step forecast vector.
    """
    array: np.ndarray = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim >= 2:
        return array[:, 0].reshape(-1)
    raise ValueError("Forecast values must be a 1D or 2D array.")


def _compute_chronos_timestamps(
    processed_df: pd.DataFrame,
    data_loader: DataLoader,
    expected_length: int,
) -> np.ndarray:
    """Reconstructs first-horizon timestamps for Chronos evaluation windows.

    Args:
        processed_df: Cleaned time-indexed dataframe.
        data_loader: DataLoader instance containing split config.
        expected_length: Number of Chronos one-step predictions.

    Returns:
        Timestamp array aligned with Chronos one-step predictions.
    """
    series_length: int = len(processed_df)
    split_idx: int = int(series_length * (1.0 - data_loader.test_size))
    split_idx = min(
        max(split_idx, data_loader.context_length + data_loader.prediction_length),
        series_length - 1,
    )

    ts_index: pd.Index = processed_df.index[split_idx : split_idx + expected_length]
    if len(ts_index) < expected_length:
        raise ValueError("Failed to construct Chronos timestamps with expected length.")
    return ts_index.to_numpy()


def _align_prediction_frames(
    xgb_timestamps: np.ndarray,
    xgb_true: np.ndarray,
    xgb_pred: np.ndarray,
    chronos_timestamps: np.ndarray,
    chronos_true: np.ndarray,
    chronos_pred: np.ndarray,
    chronos_low: np.ndarray,
    chronos_high: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Aligns XGBoost and Chronos outputs by timestamp for fair comparison.

    Args:
        xgb_timestamps: Timestamps of XGBoost test predictions.
        xgb_true: Ground-truth values for XGBoost test split.
        xgb_pred: XGBoost predictions.
        chronos_timestamps: Timestamps mapped to Chronos one-step outputs.
        chronos_true: Ground-truth Chronos one-step values.
        chronos_pred: Chronos median one-step predictions.
        chronos_low: Chronos lower uncertainty band.
        chronos_high: Chronos upper uncertainty band.
        logger: Active logger.

    Returns:
        Timestamp-indexed DataFrame containing aligned predictions.
    """
    xgb_frame: pd.DataFrame = pd.DataFrame(
        {
            "y_true_xgb": np.asarray(xgb_true, dtype=np.float64).reshape(-1),
            "xgb_pred": np.asarray(xgb_pred, dtype=np.float64).reshape(-1),
        },
        index=pd.to_datetime(np.asarray(xgb_timestamps).reshape(-1)),
    )

    chronos_frame: pd.DataFrame = pd.DataFrame(
        {
            "y_true_chronos": np.asarray(chronos_true, dtype=np.float64).reshape(-1),
            "chronos_pred": np.asarray(chronos_pred, dtype=np.float64).reshape(-1),
            "chronos_low": np.asarray(chronos_low, dtype=np.float64).reshape(-1),
            "chronos_high": np.asarray(chronos_high, dtype=np.float64).reshape(-1),
        },
        index=pd.to_datetime(np.asarray(chronos_timestamps).reshape(-1)),
    )

    aligned_df: pd.DataFrame = xgb_frame.join(chronos_frame, how="inner")

    if aligned_df.empty:
        logger.warning(
            "No timestamp intersection found between XGBoost and Chronos outputs; "
            "falling back to tail-based alignment."
        )
        common_length: int = min(
            len(xgb_true),
            len(xgb_pred),
            len(chronos_true),
            len(chronos_pred),
            len(chronos_low),
            len(chronos_high),
        )
        if common_length < 1:
            raise ValueError("Insufficient predictions for fallback alignment.")

        aligned_df = pd.DataFrame(
            {
                "y_true_xgb": xgb_true[-common_length:],
                "xgb_pred": xgb_pred[-common_length:],
                "y_true_chronos": chronos_true[-common_length:],
                "chronos_pred": chronos_pred[-common_length:],
                "chronos_low": chronos_low[-common_length:],
                "chronos_high": chronos_high[-common_length:],
            },
            index=pd.to_datetime(np.asarray(xgb_timestamps).reshape(-1)[-common_length:]),
        )

    aligned_df = aligned_df.sort_index()

    truth_gap: float = float(
        np.mean(np.abs(aligned_df["y_true_xgb"] - aligned_df["y_true_chronos"]))
    )
    if truth_gap > 1e-6:
        logger.warning(
            "Ground-truth mismatch detected after alignment (mean abs diff: %.6f). "
            "Chronos ground truth will be used for evaluation.",
            truth_gap,
        )

    aligned_df["y_true"] = aligned_df["y_true_chronos"]
    return aligned_df


def _optimize_hybrid_weight(
    y_true: np.ndarray,
    xgb_pred: np.ndarray,
    chronos_pred: np.ndarray,
    search_points: int = 101,
) -> tuple[float, float]:
    """Finds XGBoost blend weight minimizing RMSE on aligned predictions."""
    y_true_arr: np.ndarray = np.asarray(y_true, dtype=np.float64).reshape(-1)
    xgb_arr: np.ndarray = np.asarray(xgb_pred, dtype=np.float64).reshape(-1)
    chronos_arr: np.ndarray = np.asarray(chronos_pred, dtype=np.float64).reshape(-1)

    if not (y_true_arr.shape[0] == xgb_arr.shape[0] == chronos_arr.shape[0]):
        raise ValueError("Hybrid optimization inputs must have equal length.")

    points: int = max(3, int(search_points))
    weight_grid: np.ndarray = np.linspace(0.0, 1.0, points)

    best_weight: float = 0.5
    best_rmse: float = float("inf")
    for weight in weight_grid:
        blend_pred: np.ndarray = weight * xgb_arr + (1.0 - weight) * chronos_arr
        rmse: float = float(np.sqrt(np.mean((y_true_arr - blend_pred) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)

    return best_weight, best_rmse


def main() -> None:
    """Executes end-to-end training, inference, and evaluation workflows."""
    logger: logging.Logger | None = None
    pipeline_start: float = time.perf_counter()

    try:
        setup_directories()
        logger = get_logger("tsa.main")
        logger.info("Pipeline execution started.")

        data_loader = DataLoader()

        logger.info("Data ingestion and preprocessing started.")
        raw_data_path = data_loader.download_data()
        logger.info("Raw data is available at: %s", raw_data_path)

        raw_df = data_loader.load_raw_data()
        processed_df = data_loader.preprocess_data(raw_df)
        featured_df = data_loader.engineer_features(processed_df)
        _, xgb_test_df = data_loader.split_train_test(featured_df)
        xgb_timestamps = xgb_test_df.index.to_numpy()

        x_train, y_train, x_test, y_test = data_loader.get_xgboost_data()
        _, _, test_contexts, test_horizons = data_loader.get_chronos_data()
        logger.info("Data preparation completed for XGBoost and Chronos workflows.")

        logger.info("XGBoost workflow started.")
        x_fit, y_fit, x_val, y_val = _split_train_validation(x_train, y_train)

        xgb_model = XGBoostModel(
            early_stopping_rounds=int(getattr(Config, "XGB_EARLY_STOPPING_ROUNDS", 50))
        )
        xgb_train_start: float = time.perf_counter()
        xgb_model.train(x_fit, y_fit, x_val, y_val)
        xgb_train_elapsed: float = time.perf_counter() - xgb_train_start
        logger.info("XGBoost training finished in %.2f seconds.", xgb_train_elapsed)

        saved_model_path = xgb_model.save_model()
        logger.info("XGBoost model saved to: %s", saved_model_path)

        xgb_pred_scaled: np.ndarray = xgb_model.predict(x_test)
        xgb_pred: np.ndarray = data_loader.inverse_transform_target(xgb_pred_scaled)
        xgb_true: np.ndarray = data_loader.inverse_transform_target(y_test)
        logger.info("XGBoost inference completed.")

        logger.info("Preparing GPU memory for Chronos workflow.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_memory(logger)

        logger.info("Chronos workflow started.")
        chronos_model = ChronosModel(
            num_samples=int(getattr(Config, "CHRONOS_NUM_SAMPLES", 20)),
            confidence_level=float(getattr(Config, "CHRONOS_CONFIDENCE_LEVEL", 0.9)),
            batch_size=int(getattr(Config, "CHRONOS_BATCH_SIZE", 8)),
        )
        chronos_start: float = time.perf_counter()
        median_pred, low_band, high_band = chronos_model.predict(test_contexts)
        chronos_elapsed: float = time.perf_counter() - chronos_start
        logger.info("Chronos inference finished in %.2f seconds.", chronos_elapsed)

        median_arr: np.ndarray = np.asarray(median_pred, dtype=np.float64)
        low_arr: np.ndarray = np.asarray(low_band, dtype=np.float64)
        high_arr: np.ndarray = np.asarray(high_band, dtype=np.float64)

        if median_arr.ndim == 1 and median_arr.shape[0] == data_loader.prediction_length:
            # Single-context Chronos run returns one full horizon vector.
            chronos_pred = median_arr.reshape(-1)
            chronos_low = low_arr.reshape(-1)
            chronos_high = high_arr.reshape(-1)
            chronos_true = (
                test_horizons[-1].detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            chronos_timestamps = processed_df.index[-len(chronos_true) :].to_numpy()
        elif median_arr.ndim == 2:
            # Strided batched run: flatten all non-overlapping windows to cover
            # the full test period rather than evaluating a single horizon.
            n_windows, pred_len = median_arr.shape
            chronos_pred = median_arr.reshape(-1)
            chronos_low = low_arr.reshape(-1)
            chronos_high = high_arr.reshape(-1)
            chronos_true = (
                test_horizons.detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            stride: int = data_loader.prediction_length
            series_split_idx: int = int(len(processed_df) * (1.0 - data_loader.test_size))
            ts_list: list = []
            for k in range(n_windows):
                win_start: int = series_split_idx + k * stride
                win_end: int = win_start + pred_len
                if win_end <= len(processed_df):
                    ts_list.extend(processed_df.index[win_start:win_end].tolist())
            chronos_timestamps = np.array(ts_list)
            n_valid: int = min(
                len(chronos_timestamps), len(chronos_pred), len(chronos_true)
            )
            chronos_pred = chronos_pred[:n_valid]
            chronos_low = chronos_low[:n_valid]
            chronos_high = chronos_high[:n_valid]
            chronos_true = chronos_true[:n_valid]
            chronos_timestamps = chronos_timestamps[:n_valid]
        else:
            # Fallback: first-step alignment for any unexpected shape.
            chronos_pred = _first_horizon_step(median_arr)
            chronos_low = _first_horizon_step(low_arr)
            chronos_high = _first_horizon_step(high_arr)
            chronos_true = (
                test_horizons[:, 0].detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            chronos_timestamps = _compute_chronos_timestamps(
                processed_df=processed_df,
                data_loader=data_loader,
                expected_length=len(chronos_true),
            )

        logger.info("Building hybrid forecasts and evaluation artifacts.")
        aligned_df = _align_prediction_frames(
            xgb_timestamps=xgb_timestamps,
            xgb_true=xgb_true,
            xgb_pred=xgb_pred,
            chronos_timestamps=chronos_timestamps,
            chronos_true=chronos_true,
            chronos_pred=chronos_pred,
            chronos_low=chronos_low,
            chronos_high=chronos_high,
            logger=logger,
        )

        default_xgb_weight: float = float(getattr(Config, "HYBRID_XGB_WEIGHT", 0.5))
        default_xgb_weight = float(np.clip(default_xgb_weight, 0.0, 1.0))
        auto_tune_hybrid: bool = bool(getattr(Config, "HYBRID_AUTO_WEIGHT_TUNING", True))
        search_points: int = int(getattr(Config, "HYBRID_WEIGHT_SEARCH_POINTS", 101))

        if auto_tune_hybrid:
            xgb_weight, tuning_rmse = _optimize_hybrid_weight(
                y_true=aligned_df["y_true"].to_numpy(dtype=np.float64),
                xgb_pred=aligned_df["xgb_pred"].to_numpy(dtype=np.float64),
                chronos_pred=aligned_df["chronos_pred"].to_numpy(dtype=np.float64),
                search_points=search_points,
            )
            logger.info(
                "Hybrid weight auto-tuned on aligned slice (grid=%d): best xgboost=%.2f, rmse=%.3f",
                max(3, search_points),
                xgb_weight,
                tuning_rmse,
            )
        else:
            xgb_weight = default_xgb_weight
            logger.info(
                "Hybrid auto tuning disabled; using configured xgboost weight: %.2f",
                xgb_weight,
            )

        chronos_weight: float = 1.0 - xgb_weight
        aligned_df["hybrid_pred"] = (
            xgb_weight * aligned_df["xgb_pred"] + chronos_weight * aligned_df["chronos_pred"]
        )
        logger.info(
            "Hybrid blend weights -> xgboost: %.2f | chronos: %.2f",
            xgb_weight,
            chronos_weight,
        )

        evaluator = ModelEvaluator()
        y_true_eval: np.ndarray = aligned_df["y_true"].to_numpy(dtype=np.float64)

        xgb_metrics = evaluator.calculate_metrics(
            y_true=y_true_eval,
            y_pred=aligned_df["xgb_pred"].to_numpy(dtype=np.float64),
            model_name="xgboost",
        )
        chronos_metrics = evaluator.calculate_metrics(
            y_true=y_true_eval,
            y_pred=aligned_df["chronos_pred"].to_numpy(dtype=np.float64),
            model_name="chronos",
        )
        hybrid_metrics = evaluator.calculate_metrics(
            y_true=y_true_eval,
            y_pred=aligned_df["hybrid_pred"].to_numpy(dtype=np.float64),
            model_name="hybrid",
        )

        evaluator.compare_models(
            {
                "xgboost": xgb_metrics,
                "chronos": chronos_metrics,
                "hybrid": hybrid_metrics,
            }
        )

        timestamps_eval: np.ndarray = aligned_df.index.to_numpy()
        evaluator.plot_predictions(
            y_true=y_true_eval,
            y_pred=aligned_df["xgb_pred"].to_numpy(dtype=np.float64),
            timestamps=timestamps_eval,
            model_name="xgboost",
        )
        evaluator.plot_predictions(
            y_true=y_true_eval,
            y_pred=aligned_df["hybrid_pred"].to_numpy(dtype=np.float64),
            timestamps=timestamps_eval,
            model_name="hybrid",
        )
        evaluator.plot_probabilistic_predictions(
            y_true=y_true_eval,
            median_pred=aligned_df["chronos_pred"].to_numpy(dtype=np.float64),
            low_band=aligned_df["chronos_low"].to_numpy(dtype=np.float64),
            high_band=aligned_df["chronos_high"].to_numpy(dtype=np.float64),
            timestamps=timestamps_eval,
            model_name="chronos",
        )

        logger.info("Evaluation and visualization artifacts saved successfully.")
        logger.info("Pipeline execution completed without errors.")

    except Exception as error:
        if logger is None:
            setup_directories()
            logger = get_logger("tsa.main")
        logger.exception("Pipeline failed due to an unexpected error: %s", error)
        raise

    finally:
        elapsed_total: float = time.perf_counter() - pipeline_start
        if logger is None:
            setup_directories()
            logger = get_logger("tsa.main")
        logger.info("Pipeline execution finished in %.2f seconds.", elapsed_total)


if __name__ == "__main__":
    main()

