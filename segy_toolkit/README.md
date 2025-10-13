# SEG-Y Toolkit

Utilities for reading, visualising and processing SEG-Y (``.sgy``) seismic
volumes.  The package intentionally focuses on workflows that are commonly
requested during early data QC: metadata inspection, quick-look plots and a
simple approach for flattening profiles to the seafloor horizon.

## Features

- Pure Python SEG-Y reader compatible with revision 1 and 2 binary headers.
- Support for the most frequently used sample formats (IBM float, 32/16 bit
  integers, IEEE 32/64 bit floats).
- High level :class:`~segy_toolkit.io.SegyDataset` container that stores
  decoded headers and trace data as ``numpy`` arrays.
- Quick-look plotting helpers for amplitude images and wiggle sections using
  ``matplotlib``.
- Seafloor picking and flattening routines that operate directly on the
  :class:`~segy_toolkit.io.SegyDataset` object.

## Installation

The toolkit only depends on ``numpy`` and ``matplotlib``.  Install the package
and its dependencies via ``pip``:

```bash
pip install numpy matplotlib
```

To integrate it into an existing project simply copy the ``segy_toolkit``
folder into your source tree or package it as needed.

## Quick start

```python
from segy_toolkit import read_segy, plot_amplitude_image, flatten_to_seafloor

# Load SEG-Y file (revision is detected automatically)
dataset = read_segy("line.sgy")

# Visualise the section
plot_amplitude_image(dataset)

# Estimate and flatten the seafloor horizon
flattened, picks = flatten_to_seafloor(dataset)
```

The ``flatten_to_seafloor`` function returns a new
:class:`~segy_toolkit.io.SegyDataset` instance with the shifted traces alongside
the detected horizon picks.  The ``SeafloorPick`` data class stores both the
sample indices and the times in seconds.

## Algorithm notes

The seafloor picking algorithm performs a moving average smoothing step on the
absolute amplitudes, applies an adaptive threshold and chooses the shallowest
strong reflector.  This strategy works well for single reflector seafloor
profiles.  For more challenging datasets pass ``prefer_shallow=False`` to pick
the strongest reflector within an optional time window.

Trace flattening is accomplished through linear interpolation in sample space.
When ``extrapolate=False`` the algorithm leaves samples outside the valid
interval untouched which is helpful when post-processing muted data.

## Limitations

- SEG-Y sample format codes that are rarely used in practice (7, 10, 11, 12 and
  15) currently raise ``NotImplementedError``.
- Coordinate scalars in the trace header are exposed to the caller but are not
  applied automatically.
- The reader pads traces with zeros to the maximum trace length inside the file
  to guarantee a regular 2D array.

Contributions and bug reports are welcome!
