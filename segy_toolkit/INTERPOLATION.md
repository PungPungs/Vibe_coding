# SEG-Y Interpolation Workflow

This document provides an in-depth guide to the neural interpolation workflow
implemented in ``segy_toolkit.interpolation``.  The routines are designed for
production-style trace up-sampling with visibility into both progress and
resource usage.

## Dependencies

The interpolation stack relies on a collection of machine-learning and system
monitoring libraries:

| Library | Purpose |
| ------- | ------- |
| ``torch`` | Implements the multi-layer perceptron and provides optimisers and GPU acceleration. |
| ``tqdm`` | Renders progress bars for the training and inference loops. |
| ``psutil`` | Captures CPU, RAM and (optionally) GPU utilisation snapshots that are attached to the progress bars. |

Install the dependencies alongside the base toolkit with:

```bash
pip install numpy matplotlib torch tqdm psutil
```

## High-level workflow

1. **Prepare the dataset** – Load the SEG-Y file via
   :func:`segy_toolkit.io.read_segy`.  The helper normalises the textual header
   and decodes the binary/trace headers into dedicated data classes.
2. **Train the interpolator** – Call
   :func:`segy_toolkit.interpolation.interpolate_dataset_with_ann`.  The
   function maps the trace/time coordinates onto ``[0, 1]`` and fits a
   feed-forward neural network using mean squared error loss.  tqdm progress
   bars show epoch counts while the postfix area displays utilisation metrics
   sampled by ``psutil``.
3. **Generate predictions** – After training, inference runs in batches.  A
   second progress bar tracks batch completion and continues to update the
   resource statistics for long-running jobs.
4. **Export the result** – Pass the interpolated dataset to
   :func:`segy_toolkit.io.write_segy` or call
   :func:`segy_toolkit.interpolation.interpolate_and_export` to immediately
   write a SEG-Y file.  The textual header is automatically updated with the
   ``"C PROCESSING NOTE: ANN INTERPOLATION (PYTORCH) APPLIED"`` marker so downstream users
   recognise that an ANN resampling step occurred.

## Configurable parameters

The :class:`~segy_toolkit.interpolation.NeuralTraceInterpolator` constructor and
the :func:`~segy_toolkit.interpolation.interpolate_dataset_with_ann` helper
expose multiple knobs for adapting the network to a specific survey:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| ``hidden_layers`` | ``(256, 128, 64)`` | Shape of the fully connected stack. Increase layer sizes for more complex geology or reduce them for faster training. |
| ``learning_rate`` | ``1e-3`` | Adam optimiser learning rate. |
| ``epochs`` | ``200`` | Number of training epochs reported through the progress bar. |
| ``batch_size`` | ``8192`` | Training batch size. |
| ``prediction_batch_size`` | ``65536`` | Batch size used during inference. |
| ``device`` | Auto-detected | Torch device string (``"cuda"``/``"cpu"``). Set explicitly to force GPU or CPU execution. |
| ``progress`` | ``True`` | Enables tqdm progress bars for training and inference loops. |
| ``resource_monitor`` | ``True`` | Displays CPU/RAM (and optional GPU) usage snapshots alongside the progress bars. |
| ``trace_factor`` | ``1.0`` | Trace-axis up-sampling factor. |
| ``sample_factor`` | ``1.0`` | Sample/time-axis up-sampling factor. |
| ``processing_note`` | ``"ANN INTERPOLATION (PYTORCH) APPLIED"`` | Message appended to the textual header (prefixed with ``"C PROCESSING NOTE:"``). |

All routines accept a ``random_state`` argument to produce deterministic
weights and shuffling, simplifying regression testing and comparisons between
parameter sets.

## Monitoring progress and resource usage

Progress tracking is always opt-in via the ``progress`` flag.  When enabled, the
``NeuralTraceInterpolator`` instantiates tqdm progress bars for both training
and inference stages.  The postfix of each bar is populated with CPU usage,
resident memory and (when running on CUDA) GPU memory utilisation statistics.
These are collected using ``psutil`` and ``torch.cuda`` so no external tooling
is required.

For automated pipelines the progress bars can be disabled by constructing the
interpolator with ``progress=False``.  Resource monitoring can likewise be
switched off with ``resource_monitor=False`` should the hosting environment
restrict calls to ``psutil``.

## Exporting annotated SEG-Y volumes

Calling :func:`~segy_toolkit.interpolation.interpolate_and_export` trains the
network, performs the resampling and writes a SEG-Y file in one go.  The helper

* updates the binary header with the new sample count and interval,
* resamples the trace headers, keeping spatial metadata in sync,
* injects the processing note into the textual header using the first available
  free line (or the last line if the header is full), and
* stores IEEE 32-bit floating point samples (format code ``5``) by default.

This ensures that any consumer of the generated SEG-Y can immediately recognise
that interpolation occurred and retrieve the exact message that was configured
via ``processing_note``.

## Example script

```python
from segy_toolkit import interpolate_dataset_with_ann, read_segy

dataset = read_segy("line.sgy")
interpolated = interpolate_dataset_with_ann(
    dataset,
    trace_factor=1.25,
    sample_factor=1.25,
    epochs=300,
    learning_rate=5e-4,
    device="cuda",  # fall back to CPU automatically if CUDA is unavailable
)

print("Original traces:", dataset.n_traces)
print("Interpolated traces:", interpolated.n_traces)
```

Refer to the source code of ``interpolation.py`` for additional details such as
the progress-bar implementation and resource monitoring helpers.
