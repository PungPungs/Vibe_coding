"""Neural network based interpolation utilities for SEG-Y datasets."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional, Sequence

import numpy as np

from .io import (
    SegyDataset,
    TraceHeader,
    _append_processing_note,
    _normalize_text_header,
)


class NeuralTraceInterpolator:
    """Simple fully connected neural network for trace interpolation."""

    def __init__(
        self,
        *,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (128, 128),
        learning_rate: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        layer_dims = (input_dim, *self.hidden_layers, 1)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            weight = self._rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(
                np.float32
            )
            bias = np.zeros(out_dim, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int = 400,
        batch_size: int = 8192,
    ) -> None:
        """Train the network using mean squared error loss."""

        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != self.weights[0].shape[0]:
            raise ValueError("features must be of shape (n_samples, input_dim)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("targets must contain the same number of samples")

        n_samples = X.shape[0]
        batch_size = max(1, int(batch_size))
        for _ in range(max(1, int(epochs))):
            indices = self._rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]
                activations, preactivations = self._forward(xb)
                grads_w, grads_b = self._backward(activations, preactivations, yb)
                self._apply_gradients(grads_w, grads_b)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return network predictions for ``features``."""

        X = np.asarray(features, dtype=np.float32)
        activations, _ = self._forward(X)
        return activations[-1]

    # Internal helpers -------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        preactivations: list[np.ndarray] = []
        output = X
        for weight, bias in zip(self.weights, self.biases):
            z = output @ weight + bias
            preactivations.append(z)
            output = z
            if weight is not self.weights[-1]:
                output = np.maximum(0.0, output)
            activations.append(output)
        return activations, preactivations

    def _backward(
        self,
        activations: list[np.ndarray],
        preactivations: list[np.ndarray],
        targets: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = targets.shape[0]
        if m == 0:
            raise ValueError("cannot backpropagate on an empty batch")
        grads_w: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []
        delta = (activations[-1] - targets) * (2.0 / m)
        for layer in reversed(range(len(self.weights))):
            a_prev = activations[layer]
            dz = delta
            if layer < len(self.weights) - 1:
                relu_grad = (preactivations[layer] > 0).astype(np.float32)
                dz = delta * relu_grad
            grad_w = a_prev.T @ dz
            grad_b = dz.sum(axis=0)
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            delta = dz @ self.weights[layer].T
        return grads_w, grads_b

    def _apply_gradients(
        self, grads_w: Iterable[np.ndarray], grads_b: Iterable[np.ndarray]
    ) -> None:
        for i, (gw, gb) in enumerate(zip(grads_w, grads_b)):
            self.weights[i] -= self.learning_rate * gw.astype(np.float32)
            self.biases[i] -= self.learning_rate * gb.astype(np.float32)


def interpolate_dataset_with_ann(
    dataset: SegyDataset,
    *,
    trace_factor: float = 1.0,
    sample_factor: float = 1.0,
    hidden_layers: Sequence[int] = (128, 128),
    epochs: int = 400,
    learning_rate: float = 1e-3,
    batch_size: int = 8192,
    random_state: Optional[int] = None,
    processing_note: str = "ANN INTERPOLATION APPLIED",
) -> SegyDataset:
    """Upsample a SEG-Y dataset using a small fully connected ANN."""

    if trace_factor < 1.0 or sample_factor < 1.0:
        raise ValueError("trace_factor and sample_factor must be >= 1.0")

    n_traces, n_samples = dataset.n_traces, dataset.n_samples
    target_traces = max(int(round(n_traces * trace_factor)), n_traces)
    target_samples = max(int(round(n_samples * sample_factor)), n_samples)

    traces = dataset.data.astype(np.float32)
    trace_coords = np.linspace(0.0, 1.0, n_traces, dtype=np.float32)
    sample_coords = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    trace_grid, sample_grid = np.meshgrid(
        trace_coords, sample_coords, indexing="ij"
    )
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
    )
    model.fit(features, targets, epochs=epochs, batch_size=batch_size)

    new_trace_coords = np.linspace(0.0, 1.0, target_traces, dtype=np.float32)
    new_sample_coords = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
    new_trace_grid, new_sample_grid = np.meshgrid(
        new_trace_coords, new_sample_coords, indexing="ij"
    )
    new_features = np.column_stack((new_trace_grid.ravel(), new_sample_grid.ravel()))
    predictions = model.predict(new_features).reshape(target_traces, target_samples)

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
        data=predictions.astype(np.float32),
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
    hidden_layers: Sequence[int] = (128, 128),
    epochs: int = 400,
    learning_rate: float = 1e-3,
    batch_size: int = 8192,
    random_state: Optional[int] = None,
    processing_note: str = "ANN INTERPOLATION APPLIED",
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
        random_state=random_state,
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
