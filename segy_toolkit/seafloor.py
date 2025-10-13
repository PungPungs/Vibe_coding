"""Algorithms for automatic seafloor picking and flattening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .io import SegyDataset


@dataclass
class SeafloorPick:
    """Container for seafloor horizon picks."""

    indices: np.ndarray
    times: np.ndarray
    reference_index: int


def estimate_seafloor_horizon(
    dataset: SegyDataset,
    *,
    window: Optional[Tuple[float, float]] = None,
    smooth_samples: int = 5,
    prefer_shallow: bool = True,
) -> SeafloorPick:
    """Estimate a seafloor horizon using an amplitude based picker."""

    if dataset.data.ndim != 2:
        raise ValueError("SEG-Y dataset must contain 2D data")

    amplitudes = np.abs(dataset.data)
    if smooth_samples and smooth_samples > 1:
        kernel = np.ones(smooth_samples, dtype=float) / smooth_samples
        amplitudes = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), 1, amplitudes
        )

    time_axis = dataset.time_axis
    if window is not None:
        start, end = window
        if start >= end:
            raise ValueError("window start must be smaller than window end")
        start_idx = int(np.searchsorted(time_axis, start, side="left"))
        end_idx = int(np.searchsorted(time_axis, end, side="right"))
    else:
        start_idx = 0
        end_idx = dataset.n_samples

    restricted = amplitudes[:, start_idx:end_idx]
    if restricted.size == 0:
        raise ValueError("Window does not overlap with trace time axis")

    if prefer_shallow:
        percentiles = np.percentile(restricted, 75, axis=1)
        picks_within = []
        for trace, threshold in zip(restricted, percentiles):
            candidates = np.flatnonzero(trace >= threshold)
            if candidates.size:
                picks_within.append(int(candidates[0]))
            else:
                picks_within.append(int(np.argmax(trace)))
        picks_within = np.array(picks_within, dtype=int)
    else:
        picks_within = np.argmax(restricted, axis=1)

    indices = picks_within + start_idx
    times = time_axis[indices]
    reference_index = int(np.median(indices))
    return SeafloorPick(indices=indices, times=times, reference_index=reference_index)


def flatten_to_seafloor(
    dataset: SegyDataset,
    *,
    picks: Optional[SeafloorPick] = None,
    window: Optional[Tuple[float, float]] = None,
    smooth_samples: int = 5,
    prefer_shallow: bool = True,
    extrapolate: bool = True,
) -> Tuple[SegyDataset, SeafloorPick]:
    """Flatten a section so that the seafloor horizon becomes horizontal."""

    if picks is None:
        picks = estimate_seafloor_horizon(
            dataset,
            window=window,
            smooth_samples=smooth_samples,
            prefer_shallow=prefer_shallow,
        )

    flattened = np.zeros_like(dataset.data)
    ref = picks.reference_index
    sample_indices = np.arange(dataset.n_samples, dtype=float)

    for i in range(dataset.n_traces):
        shift = picks.indices[i] - ref
        if shift == 0:
            flattened[i] = dataset.data[i]
            continue

        shifted_index = sample_indices + shift
        if extrapolate:
            flattened[i] = np.interp(
                sample_indices,
                shifted_index,
                dataset.data[i],
                left=dataset.data[i, 0],
                right=dataset.data[i, -1],
            )
        else:
            valid = (shifted_index >= 0) & (shifted_index < dataset.n_samples)
            if np.any(valid):
                flattened[i, valid] = np.interp(
                    sample_indices[valid],
                    shifted_index[valid],
                    dataset.data[i, valid],
                )

    return dataset.with_data(flattened), picks
