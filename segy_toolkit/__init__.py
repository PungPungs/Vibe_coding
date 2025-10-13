"""Utilities for working with SEG-Y (SGY) seismic data."""

from .io import BinaryHeader, SegyDataset, SegyReader, TraceHeader, read_segy
from .visualization import plot_amplitude_image, plot_wiggle_section
from .seafloor import estimate_seafloor_horizon, flatten_to_seafloor

__all__ = [
    "SegyDataset",
    "SegyReader",
    "BinaryHeader",
    "TraceHeader",
    "read_segy",
    "plot_amplitude_image",
    "plot_wiggle_section",
    "estimate_seafloor_horizon",
    "flatten_to_seafloor",
]
