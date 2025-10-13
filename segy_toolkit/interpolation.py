"""Neural network based interpolation utilities for SEG-Y datasets."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Iterable, Optional, Sequence

import numpy as np

from .io import (
    SegyDataset,
    TraceHeader,
    _append_processing_note,
    _normalize_text_header,
)

try:  # pragma: no cover - dependency availability is runtime specific
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - dependency availability is runtime specific
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    tqdm = None  # type: ignore[assignment]
    _TQDM_AVAILABLE = False

try:  # pragma: no cover - dependency availability is runtime specific
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


class NeuralTraceInterpolator:
    """PyTorch based multi-layer perceptron for trace interpolation.

    Parameters
    ----------
    input_dim:
        Dimensionality of the coordinate input. The default value (``2``)
        corresponds to ``(trace_index, sample_index)`` features.
    hidden_layers:
        Sizes of the hidden layers. Each entry creates a fully connected layer
        followed by a ReLU activation.
    learning_rate:
        Optimiser learning rate.
    device:
        Optional torch device string. When ``None`` (default) the class selects
        ``"cuda"`` if a GPU is available, otherwise ``"cpu"``.
    random_state:
        Optional seed that is forwarded to both ``torch`` and ``numpy`` for
        reproducible training.
    progress:
        When ``True`` (default) tqdm based progress bars are displayed for both
        training and inference.
    resource_monitor:
        When ``True`` (default) CPU, RAM and GPU (if available) usage snapshots
        are captured and displayed alongside the progress bar.
    dtype:
        Torch dtype used during training and inference.
    """

    def __init__(
        self,
        *,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (256, 128, 64),
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        progress: bool = True,
        resource_monitor: bool = True,
        dtype: "torch.dtype" = None,  # type: ignore[assignment]
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "NeuralTraceInterpolator requires PyTorch. Install it via"
                " 'pip install torch'."
            )
        if progress and not _TQDM_AVAILABLE:
            raise ImportError(
                "Progress reporting requires tqdm. Install it via"
                " 'pip install tqdm'."
            )
        if resource_monitor and not _PSUTIL_AVAILABLE:
            raise ImportError(
                "Resource monitoring requires psutil. Install it via"
                " 'pip install psutil'."
            )

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.learning_rate = float(learning_rate)
        self.progress = bool(progress)
        self.resource_monitor = bool(resource_monitor)

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():  # pragma: no cover - hardware specific
                torch.cuda.manual_seed_all(random_state)

        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.dtype = dtype or torch.float32

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device, dtype=self.dtype)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int = 200,
        batch_size: int = 8192,
        shuffle: bool = True,
    ) -> None:
        """Train the network using mean squared error loss."""

        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != self.model[0].in_features:
            raise ValueError("features must be of shape (n_samples, input_dim)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("targets must contain the same number of samples")
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        tensor_x = torch.from_numpy(X).to(self.device, dtype=self.dtype)
        tensor_y = torch.from_numpy(y).to(self.device, dtype=self.dtype)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        progress_bar = _create_progress_bar(
            range(epochs),
            enabled=self.progress,
            description="Training",
            unit="epoch",
        )

        process = None
        if self.resource_monitor and _PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            process.cpu_percent(interval=None)  # prime measurement

        for epoch in progress_bar:
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach().item() * batch_x.size(0)

            epoch_loss = running_loss / len(dataset)
            metrics = {"loss": f"{epoch_loss:.4e}"}
            metrics.update(_resource_snapshot(process, self.device))
            progress_bar.set_postfix(metrics)

        progress_bar.close()
        self._fitted = True

    def predict(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Return network predictions for ``features``."""

        if not self._fitted:
            raise RuntimeError("The model must be fitted before calling predict().")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        X = np.asarray(features, dtype=np.float32)
        tensor_x = torch.from_numpy(X).to(self.device, dtype=self.dtype)
        loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)

        predictions: list[np.ndarray] = []
        progress_bar = _create_progress_bar(
            loader,
            enabled=self.progress,
            description="Inference",
            unit="batch",
        )

        process = None
        if self.resource_monitor and _PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            process.cpu_percent(interval=None)  # prime measurement

        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in progress_bar:
                outputs = self.model(batch_x)
                predictions.append(outputs.detach().cpu().numpy())
                metrics = _resource_snapshot(process, self.device)
                progress_bar.set_postfix(metrics)

        progress_bar.close()
        return np.vstack(predictions)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def to(self, device: str) -> "NeuralTraceInterpolator":
        """Move the underlying model to ``device``."""

        self.device = torch.device(device)
        self.model.to(self.device, dtype=self.dtype)
        return self


def interpolate_dataset_with_ann(
    dataset: SegyDataset,
    *,
    trace_factor: float = 1.0,
    sample_factor: float = 1.0,
    hidden_layers: Sequence[int] = (256, 128, 64),
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 8192,
    prediction_batch_size: int = 65536,
    random_state: Optional[int] = None,
    device: Optional[str] = None,
    progress: bool = True,
    resource_monitor: bool = True,
    processing_note: str = "ANN INTERPOLATION (PYTORCH) APPLIED",
) -> SegyDataset:
    """Upsample a SEG-Y dataset using a PyTorch based neural interpolator."""

    if trace_factor < 1.0 or sample_factor < 1.0:
        raise ValueError("trace_factor and sample_factor must be >= 1.0")

    n_traces, n_samples = dataset.n_traces, dataset.n_samples
    target_traces = max(int(round(n_traces * trace_factor)), n_traces)
    target_samples = max(int(round(n_samples * sample_factor)), n_samples)

    traces = dataset.data.astype(np.float32)
    trace_coords = np.linspace(0.0, 1.0, n_traces, dtype=np.float32)
    sample_coords = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    trace_grid, sample_grid = np.meshgrid(trace_coords, sample_coords, indexing="ij")
    features = np.column_stack((trace_grid.ravel(), sample_grid.ravel()))
    targets = traces.reshape(-1, 1)

    mask = np.isfinite(targets[:, 0])
    if not np.any(mask):
        raise ValueError("dataset does not contain finite samples for training")
    features = features[mask]
    targets = targets[mask]

    model = NeuralTraceInterpolator(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        random_state=random_state,
        device=device,
        progress=progress,
        resource_monitor=resource_monitor,
    )
    model.fit(features, targets, epochs=epochs, batch_size=batch_size)

    new_trace_coords = np.linspace(0.0, 1.0, target_traces, dtype=np.float32)
    new_sample_coords = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
    new_trace_grid, new_sample_grid = np.meshgrid(
        new_trace_coords, new_sample_coords, indexing="ij"
    )
    new_features = np.column_stack((new_trace_grid.ravel(), new_sample_grid.ravel()))
    predictions = model.predict(new_features, batch_size=prediction_batch_size)
    prediction_grid = predictions.reshape(target_traces, target_samples)

    if target_samples == n_samples:
        new_sample_interval_us = dataset.sample_interval_us
    else:
        new_sample_interval_us = max(
            1, int(round(dataset.sample_interval_us / sample_factor))
        )

    new_trace_headers = _resample_trace_headers(
        dataset.trace_headers,
        target_traces=target_traces,
        target_samples=target_samples,
        sample_interval_us=new_sample_interval_us,
        endianness=dataset.binary_header.endianness,
    )
    new_binary_header = dataset.binary_header.updated(
        samples_per_trace=target_samples,
        sample_interval_us=new_sample_interval_us,
        data_sample_format=5,
    )
    new_text_header = _append_processing_note(
        dataset.text_header, processing_note
    )

    return replace(
        dataset,
        data=prediction_grid.astype(np.float32),
        binary_header=new_binary_header,
        trace_headers=new_trace_headers,
        text_header=_normalize_text_header(new_text_header),
        sample_interval_us=new_sample_interval_us,
    )


def interpolate_and_export(
    dataset: SegyDataset,
    path: str,
    *,
    trace_factor: float = 1.0,
    sample_factor: float = 1.0,
    hidden_layers: Sequence[int] = (256, 128, 64),
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 8192,
    prediction_batch_size: int = 65536,
    random_state: Optional[int] = None,
    device: Optional[str] = None,
    progress: bool = True,
    resource_monitor: bool = True,
    processing_note: str = "ANN INTERPOLATION (PYTORCH) APPLIED",
) -> SegyDataset:
    """Interpolate ``dataset`` and immediately write it to ``path``."""

    interpolated = interpolate_dataset_with_ann(
        dataset,
        trace_factor=trace_factor,
        sample_factor=sample_factor,
        hidden_layers=hidden_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        prediction_batch_size=prediction_batch_size,
        random_state=random_state,
        device=device,
        progress=progress,
        resource_monitor=resource_monitor,
        processing_note=processing_note,
    )

    from .io import write_segy

    write_segy(interpolated, path)
    return interpolated


def _resample_trace_headers(
    headers: Sequence[TraceHeader],
    *,
    target_traces: int,
    target_samples: int,
    sample_interval_us: int,
    endianness: str,
) -> list[TraceHeader]:
    if not headers:
        raise ValueError("SEG-Y dataset does not contain trace headers")

    prepared: list[TraceHeader] = []
    src_count = len(headers)
    for idx in range(target_traces):
        if src_count == 1:
            template = headers[0]
        else:
            src_idx = int(round(idx * (src_count - 1) / max(target_traces - 1, 1)))
            template = headers[src_idx]
        prepared.append(
            template.updated(
                endianness=endianness,
                trace_sequence_line=idx + 1,
                trace_sequence_file=idx + 1,
                trace_number_within_field_record=idx + 1,
                trace_number_within_ensemble=idx + 1,
                samples_in_trace=target_samples,
                sample_interval_us=sample_interval_us,
            )
        )
    return list(prepared)


def _create_progress_bar(
    iterable: Iterable,
    *,
    enabled: bool,
    description: str,
    unit: str,
):
    if not enabled:
        return _NoOpProgress(iterable)
    if not _TQDM_AVAILABLE:
        raise RuntimeError("tqdm is required for progress tracking")
    return tqdm(iterable, desc=description, unit=unit)


def _resource_snapshot(process, device) -> dict[str, str]:
    stats: dict[str, str] = {}
    if process is not None:
        cpu = process.cpu_percent(interval=None)
        mem = process.memory_info().rss / (1024**2)
        stats["cpu%"] = f"{cpu:.1f}"
        stats["ram_mb"] = f"{mem:.1f}"
    if _TORCH_AVAILABLE and isinstance(device, torch.device):
        if device.type == "cuda" and torch.cuda.is_available():  # pragma: no cover
            mem_gpu = torch.cuda.memory_allocated(device) / (1024**2)
            stats["gpu_mem_mb"] = f"{mem_gpu:.1f}"
    return stats


class _NoOpProgress:
    """Fallback progress handler used when progress bars are disabled."""

    def __init__(self, iterable: Iterable) -> None:
        self._iterable = iterable

    def __iter__(self):
        yield from self._iterable

    def set_postfix(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None

