# SEG-Y Toolkit

Utilities for reading, visualising and processing SEG-Y (``.sgy``) seismic
volumes.  The package intentionally focuses on workflows that are commonly
requested during early data QC: metadata inspection, quick-look plots and a
simple approach for flattening profiles to the seafloor horizon.

The code in this folder is split across three modules:

| Module | Purpose |
| ------ | ------- |
| ``io.py`` | Low level reader that loads SEG-Y revision 1/2 files into the structured :class:`~segy_toolkit.io.SegyDataset` container.  It exposes :class:`~segy_toolkit.io.BinaryHeader` and :class:`~segy_toolkit.io.TraceHeader` data classes so header fields can be inspected programmatically. |
| ``visualization.py`` | ``matplotlib`` based helpers for amplitude images and wiggle plots, ideal for quick QC figures. |
| ``seafloor.py`` | Automatic seafloor picking (:func:`~segy_toolkit.seafloor.estimate_seafloor_horizon`) and flattening (:func:`~segy_toolkit.seafloor.flatten_to_seafloor`) routines. |
| ``interpolation.py`` | Neural network driven interpolation utilities that can up-sample volumes and export SEG-Y files annotated with processing notes. |

All modules are re-exported from ``segy_toolkit.__init__`` so the most common
symbols can be imported directly via ``from segy_toolkit import ...``.

## Features

- Pure Python SEG-Y reader compatible with revision 1 and 2 binary headers and
  both big-endian and little-endian trace payloads.
- Support for the most frequently used sample formats (IBM float, 32/16 bit
  integers, IEEE 32/64 bit floats).
- High level :class:`~segy_toolkit.io.SegyDataset` container that stores
  decoded headers and trace data as ``numpy`` arrays.
- Quick-look plotting helpers for amplitude images and wiggle sections using
  ``matplotlib`` that automatically display the figure (set ``show=False`` to
  reuse axes in scripted workflows).
- Seafloor picking and flattening routines that operate directly on the
  :class:`~segy_toolkit.io.SegyDataset` object.
- Neural network based interpolation that can densify the trace/time grid and
  persist the result back to SEG-Y with a processing note embedded in the
  textual header.
- A light-weight :func:`~segy_toolkit.io.write_segy` helper for exporting
  processed datasets back to disk.
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
from pathlib import Path

from segy_toolkit import (
    BinaryHeader,
    SegyDataset,
    flatten_to_seafloor,
    interpolate_and_export,
    plot_amplitude_image,
    read_segy,
)

import numpy as np

# Load SEG-Y file (revision and byte order are detected automatically)
dataset: SegyDataset = read_segy("line.sgy")

# Inspect metadata
binary_header: BinaryHeader = dataset.binary_header
print("Samples per trace:", binary_header.samples_per_trace)
print("Data format code:", binary_header.data_sample_format)

# Trace specific metadata is exposed through TraceHeader objects
first_trace_header = dataset.trace_headers[0]
print("First trace inline number:", first_trace_header.trace_sequence_line)

# Visualise the section
plot_amplitude_image(dataset)

# Estimate and flatten the seafloor horizon
flattened, picks = flatten_to_seafloor(dataset)
print("Reference trace index:", picks.reference_index)

# Persist the flattened section if required
np.save(Path("flattened.npy"), flattened.data)

# Upsample the dataset and store a SEG-Y copy with a processing note
interpolated = interpolate_and_export(
    dataset,
    "line_interpolated.sgy",
    trace_factor=1.5,
    sample_factor=1.25,
)
print("Interpolated trace count:", interpolated.n_traces)
```

The ``flatten_to_seafloor`` function returns a new
:class:`~segy_toolkit.io.SegyDataset` instance with the shifted traces alongside
the detected horizon picks.  The ``SeafloorPick`` data class stores both the
sample indices and the times in seconds.  Use :meth:`~segy_toolkit.io.SegyDataset.trace_header_dicts`
or :meth:`~segy_toolkit.io.SegyDataset.binary_header_dict` when a dictionary
representation of the headers is preferred (e.g. for serialisation).

## Algorithm notes

The seafloor picking algorithm performs a moving average smoothing step on the
absolute amplitudes, applies an adaptive threshold and chooses the shallowest
strong reflector.  This strategy works well for single reflector seafloor
profiles.  For more challenging datasets pass ``prefer_shallow=False`` to pick
the strongest reflector within an optional time window.

Trace flattening is accomplished through linear interpolation in sample space.
When ``extrapolate=False`` the algorithm leaves samples outside the valid
interval untouched which is helpful when post-processing muted data.
The picker works entirely on ``numpy`` arrays so the results are deterministic
and easily reproducible.

The neural interpolation pipeline represents the amplitude field as a function
of normalised trace/time coordinates and trains a compact fully connected
network to minimise the mean squared error on available samples.  The fitted
network is then evaluated on a denser grid, allowing trace/time up-sampling
without requiring third-party machine learning frameworks.  The helper
:func:`~segy_toolkit.interpolation.interpolate_and_export` automatically embeds
``"C PROCESSING NOTE: ANN INTERPOLATION APPLIED"`` in the textual header of the
generated SEG-Y file so downstream users can identify the interpolated volume.

## Limitations

- SEG-Y sample format codes that are rarely used in practice (7, 10, 11, 12 and
  15) currently raise ``NotImplementedError``.
- Coordinate scalars in the trace header are exposed to the caller but are not
  applied automatically.
- The reader pads traces with zeros to the maximum trace length inside the file
  to guarantee a regular 2D array.

### Running the examples

Create a virtual environment, install ``numpy``/``matplotlib`` and run any of
the snippets from above using the Python interpreter:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python - <<'PY'
from pathlib import Path

import numpy as np

from segy_toolkit import flatten_to_seafloor, plot_amplitude_image, read_segy

dataset = read_segy("line.sgy")
plot_amplitude_image(dataset)
flattened, picks = flatten_to_seafloor(dataset)
np.save(Path("flattened.npy"), flattened.data)
print("Reference trace index:", picks.reference_index)
PY
```

Alternatively, copy the longer example from the *Quick start* section into your
own script or notebook.

Contributions and bug reports are welcome!
