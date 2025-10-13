"""Visualization helpers for SEG-Y datasets."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .io import SegyDataset


def plot_amplitude_image(
    dataset: SegyDataset,
    *,
    cmap: str = "seismic",
    clip_percentile: float = 99.0,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a seismic section using :func:`matplotlib.pyplot.imshow`.

    Parameters
    ----------
    dataset:
        Dataset containing the traces.
    cmap:
        Matplotlib colormap.
    clip_percentile:
        Amplitude clipping value (symmetrical).  Set to ``None`` to disable
        clipping.
    figsize:
        Optional figure size in inches when ``ax`` is not supplied.
    ax:
        Optional axes object to reuse.
    show:
        When ``True`` call :func:`matplotlib.pyplot.show` so the plot is
        displayed immediately.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = dataset.data.copy()
    if clip_percentile is not None:
        clip = np.percentile(np.abs(data), clip_percentile)
        if clip > 0:
            data = np.clip(data, -clip, clip)

    extent = [0, dataset.n_traces, dataset.time_axis[-1], dataset.time_axis[0]]
    ax.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        extent=extent,
    )
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time [s]")
    ax.set_title("Amplitude section")
    fig.tight_layout()
    if show:
        plt.show()
    return ax


def plot_wiggle_section(
    dataset: SegyDataset,
    *,
    trace_indices: Optional[Iterable[int]] = None,
    scale: float = 1.0,
    color: str = "black",
    fill_positive: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a wiggle-trace seismic section.

    Parameters
    ----------
    dataset:
        Dataset containing the traces to display.
    trace_indices:
        Optional subset of trace indices to plot.
    scale:
        Scalar applied to the trace amplitudes.
    color:
        Line colour for the wiggles and filled area.
    fill_positive:
        When ``True`` fill positive lobes to mimic classic seismic plots.
    ax:
        Optional axes object to reuse.
    figsize:
        Figure size when ``ax`` is not supplied.
    show:
        When ``True`` call :func:`matplotlib.pyplot.show` so the plot is
        displayed immediately.
    """
) -> plt.Axes:
    """Plot a wiggle-trace seismic section."""

    traces = dataset.data
    time_axis = dataset.time_axis

    if trace_indices is None:
        trace_indices = range(dataset.n_traces)
    trace_indices = list(trace_indices)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i in trace_indices:
        if i < 0 or i >= dataset.n_traces:
            raise IndexError(f"Trace index {i} outside of dataset range")
        trace = traces[i] * scale
        offset = i
        ax.plot(trace + offset, time_axis, color=color, linewidth=0.5)
        if fill_positive:
            ax.fill_betweenx(
                time_axis,
                offset,
                trace + offset,
                where=(trace + offset) > offset,
                color=color,
                alpha=0.4,
            )

    ax.invert_yaxis()
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time [s]")
    ax.set_title("Wiggle section")
    fig.tight_layout()
    if show:
        plt.show()
    return ax
