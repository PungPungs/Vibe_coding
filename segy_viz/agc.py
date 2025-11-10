"""Automatic gain control utilities."""

from __future__ import annotations

import numpy as np
from enum import Enum


class GainControl(str, Enum):
    RMS = "rms"
    MAX = "max"


class AGC:
    """Automatic gain control operations."""

    @staticmethod
    def apply(
        data: np.ndarray,
        window: int,
        method: GainControl | str = GainControl.RMS,
    ) -> np.ndarray:
        if window <= 1 or data.size == 0:
            return data

        if not isinstance(method, GainControl):
            try:
                method = GainControl(str(method).lower())
            except ValueError:
                method = GainControl.RMS

        window = max(int(window), 3)
        if window % 2 == 0:
            window += 1

        half = window // 2
        padded = np.pad(data, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window

        gain_map = np.empty_like(data, dtype=np.float32)

        if method == GainControl.RMS:
            source = np.square(padded, dtype=np.float32)
            for idx in range(data.shape[1]):
                conv = np.convolve(source[:, idx], kernel, mode="valid")
                gain_map[:, idx] = np.sqrt(conv, dtype=np.float32)
        else:
            source = np.abs(padded, dtype=np.float32)
            for idx in range(data.shape[1]):
                conv = np.convolve(source[:, idx], kernel, mode="valid")
                gain_map[:, idx] = conv

        gain_map = np.clip(gain_map, 1e-6, None)
        return data / gain_map
