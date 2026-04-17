"""Chronos foundation-model wrapper for probabilistic time series forecasting.

Chronos models tokenize numeric histories into a sequence representation,
similar to language models that process text tokens. This lets the model
generate multiple plausible future trajectories instead of a single point
estimate.
"""

from __future__ import annotations

from typing import Any, TypeAlias, Union

import numpy as np
import torch

try:
    from chronos import ChronosPipeline  # pyright: ignore[reportMissingImports]  # type: ignore[import-not-found]
except ImportError:
    from transformers import ChronosPipeline  # pyright: ignore[reportMissingImports]  # type: ignore[import-not-found]

try:
    from src.config import Config
except ImportError:
    from src import config as _config

    class Config:  # type: ignore[too-many-instance-attributes]
        """Compatibility layer when src.config exposes module-level constants."""

        CHRONOS_MODEL_ID: str = str(
            getattr(_config, "CHRONOS_MODEL_ID", "amazon/chronos-t5-base")
        )
        DEVICE: torch.device | str = getattr(_config, "DEVICE", torch.device("cpu"))
        CONTEXT_LENGTH: int = int(getattr(_config, "CONTEXT_LENGTH", 512))
        PREDICTION_LENGTH: int = int(getattr(_config, "PREDICTION_LENGTH", 24))


ContextInput: TypeAlias = Union[torch.Tensor, list[torch.Tensor], np.ndarray]
ForecastOutput: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


class ChronosModel:
    """Runs Chronos probabilistic forecasts under strict VRAM constraints.

    The class enforces `torch.bfloat16` during model loading to reduce memory
    pressure on 8GB GPUs while keeping throughput high enough for iterative
    experimentation.
    """

    def __init__(
        self,
        config: type[Config] = Config,
        num_samples: int = 20,
        confidence_level: float = 0.9,
    ) -> None:
        """Initializes Chronos pipeline with hardware-aware defaults.

        Args:
            config: Configuration source class from src.config.
            num_samples: Number of trajectories sampled per forecast horizon.
            confidence_level: Central interval probability, e.g. 0.8 or 0.9.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError("confidence_level must be in the (0, 1) interval.")

        self.config: type[Config] = config
        self.model_id: str = str(getattr(config, "CHRONOS_MODEL_ID"))
        self.device_map: torch.device | str = getattr(config, "DEVICE")
        self.context_length: int = int(getattr(config, "CONTEXT_LENGTH", 512))
        self.prediction_length: int = int(getattr(config, "PREDICTION_LENGTH"))
        self.num_samples: int = num_samples
        self.confidence_level: float = confidence_level
        self.torch_dtype: torch.dtype = torch.bfloat16

        self.pipeline: ChronosPipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> ChronosPipeline:
        """Builds Chronos pipeline with bfloat16 to minimize VRAM usage.

        bfloat16 is mandatory here because Chronos relies on transformer-style
        tensor operations whose activation memory scales quickly with context and
        sample count; reduced precision keeps inference stable on 8GB cards.
        """
        try:
            return ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        except (TypeError, ValueError):
            normalized_device: str = (
                self.device_map.type
                if isinstance(self.device_map, torch.device)
                else str(self.device_map)
            )
            return ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=normalized_device,
                torch_dtype=torch.bfloat16,
            )

    @staticmethod
    def _to_1d_tensor(series: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Converts a sequence to a clean 1D float tensor.

        Args:
            series: Input context sequence.

        Returns:
            One-dimensional tensor representation.
        """
        tensor: torch.Tensor
        if isinstance(series, torch.Tensor):
            tensor = series.detach().clone().to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(series, dtype=torch.float32)

        tensor = tensor.squeeze()
        if tensor.ndim != 1:
            raise ValueError("Each context sequence must be one-dimensional after squeeze.")
        return tensor

    def _prepare_context(self, context_series: ContextInput) -> torch.Tensor | list[torch.Tensor]:
        """Normalizes inference input into Chronos-compatible context format.

        Chronos expects target-only 1D histories. For batched input this method
        returns either a stacked 2D tensor (equal lengths) or a list of 1D
        tensors (variable lengths).

        Args:
            context_series: Raw context input.

        Returns:
            Prepared context accepted by ChronosPipeline.predict.
        """
        if isinstance(context_series, np.ndarray):
            if context_series.ndim == 1:
                return self._to_1d_tensor(context_series).unsqueeze(0)
            if context_series.ndim == 2:
                return torch.as_tensor(context_series, dtype=torch.float32)
            raise ValueError("NumPy context input must be 1D or 2D.")

        if isinstance(context_series, torch.Tensor):
            if context_series.ndim == 1:
                return self._to_1d_tensor(context_series).unsqueeze(0)
            if context_series.ndim == 2:
                return context_series.detach().clone().to(dtype=torch.float32)
            raise ValueError("Tensor context input must be 1D or 2D.")

        if isinstance(context_series, list):
            if not context_series:
                raise ValueError("context_series list cannot be empty.")

            normalized_list: list[torch.Tensor] = [
                self._to_1d_tensor(item) for item in context_series
            ]
            unique_lengths: set[int] = {tensor.shape[0] for tensor in normalized_list}
            if len(unique_lengths) == 1:
                return torch.stack(normalized_list, dim=0)
            return normalized_list

        raise TypeError("Unsupported context_series type.")

    @staticmethod
    def _ensure_sample_axis(samples: torch.Tensor) -> torch.Tensor:
        """Normalizes forecast samples into [batch, num_samples, horizon] format."""
        if samples.ndim == 1:
            return samples.unsqueeze(0).unsqueeze(0)
        if samples.ndim == 2:
            return samples.unsqueeze(0)
        if samples.ndim == 3:
            return samples
        raise ValueError("Unexpected forecast output shape from Chronos pipeline.")

    def _compute_prediction_bands(self, samples: torch.Tensor) -> ForecastOutput:
        """Computes median forecast and confidence intervals from trajectories.

        Args:
            samples: Forecast trajectories in [batch, num_samples, horizon] format.

        Returns:
            Tuple of (median_forecast, low_band, high_band) as NumPy arrays.
        """
        samples = samples.to(dtype=torch.float32)

        alpha: float = 1.0 - self.confidence_level
        low_q: float = alpha / 2.0
        high_q: float = 1.0 - low_q

        median_forecast: torch.Tensor = torch.median(samples, dim=1).values
        low_band: torch.Tensor = torch.quantile(samples, q=low_q, dim=1)
        high_band: torch.Tensor = torch.quantile(samples, q=high_q, dim=1)

        median_np: np.ndarray = median_forecast.detach().cpu().numpy()
        low_np: np.ndarray = low_band.detach().cpu().numpy()
        high_np: np.ndarray = high_band.detach().cpu().numpy()

        if median_np.shape[0] == 1:
            return median_np[0], low_np[0], high_np[0]
        return median_np, low_np, high_np

    def _truncate_context_for_vram(
        self,
        prepared_context: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        """Limits context length to reduce activation memory on constrained GPUs."""
        max_context: int = max(1, int(self.context_length))

        if isinstance(prepared_context, torch.Tensor):
            if prepared_context.ndim == 1:
                return prepared_context[-max_context:]
            if prepared_context.ndim == 2:
                # Use only the latest context window to keep inference within VRAM limits.
                return prepared_context[-1:, -max_context:]
            raise ValueError("Prepared tensor context must be 1D or 2D.")

        if isinstance(prepared_context, list):
            return [prepared_context[-1][-max_context:]]

        raise TypeError("Unsupported prepared context type.")

    def _predict_single_context(self, context: torch.Tensor) -> torch.Tensor:
        """Runs Chronos prediction for a single context window."""
        try:
            raw_forecast: Any = self.pipeline.predict(
                context,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
                limit_prediction_length=False,
            )
        except TypeError:
            raw_forecast = self.pipeline.predict(
                context,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
            )

        forecast_samples: torch.Tensor = torch.as_tensor(raw_forecast)
        return self._ensure_sample_axis(forecast_samples)

    def predict(self, context_series: ContextInput) -> ForecastOutput:
        """Generates probabilistic Chronos forecasts from historical context.

        Chronos treats numeric sequences as discrete token streams and samples
        multiple future trajectories. This method aggregates those trajectories
        into a robust central forecast and uncertainty bands.

        Args:
            context_series: 1D target history (single or batch form).

        Returns:
            Tuple containing median forecast, lower band, and upper band.
        """
        prepared_context: torch.Tensor | list[torch.Tensor] = self._prepare_context(
            context_series
        )
        prepared_context = self._truncate_context_for_vram(prepared_context)

        if isinstance(prepared_context, torch.Tensor) and prepared_context.ndim == 2:
            if prepared_context.shape[0] == 1:
                forecast_samples = self._predict_single_context(prepared_context)
            else:
                sample_batches: list[torch.Tensor] = []
                for row_idx in range(prepared_context.shape[0]):
                    single_context: torch.Tensor = prepared_context[row_idx].unsqueeze(0)
                    sample_batches.append(self._predict_single_context(single_context))
                forecast_samples = torch.cat(sample_batches, dim=0)
        elif isinstance(prepared_context, list):
            sample_batches = []
            for context in prepared_context:
                sample_batches.append(self._predict_single_context(context.unsqueeze(0)))
            forecast_samples = torch.cat(sample_batches, dim=0)
        else:
            if isinstance(prepared_context, torch.Tensor):
                forecast_samples = self._predict_single_context(prepared_context)
            else:
                raise TypeError("Unsupported prepared context type.")

        return self._compute_prediction_bands(forecast_samples)

