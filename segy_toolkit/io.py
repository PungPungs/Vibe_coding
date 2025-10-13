"""SEG-Y (SGY) reading utilities.

The goal of this module is not to be a drop-in replacement for mature
libraries such as :mod:`obspy` or :mod:`segyio`, but instead to provide a
self-contained reader that covers the most common SEG-Y use cases.  The
reader understands revision 1 and 2 binary headers, several data sample
formats and produces :class:`SegyDataset` objects that expose both the
raw numpy data and the decoded headers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional
import codecs
import io
import os
import struct

import numpy as np

_TEXT_HEADER_SIZE = 3200
_BINARY_HEADER_SIZE = 400
_TRACE_HEADER_SIZE = 240


@dataclass(frozen=True)
class SegyDataset:
    """Container object storing SEG-Y traces and metadata."""

    text_header: str
    binary_header: Dict[str, int | bytes]
    trace_headers: List[Dict[str, int | bytes]]
    data: np.ndarray
    sample_interval_us: int
    revision: int
    path: Optional[Path] = None

    def with_data(self, data: np.ndarray) -> "SegyDataset":
        """Return a copy of the dataset with ``data`` replaced."""

        if data.shape != self.data.shape:
            raise ValueError(
                "New data must have the same shape as the existing traces"
            )
        return replace(self, data=data)

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def time_axis(self) -> np.ndarray:
        """Return the time/depth axis in seconds."""

        dt = self.sample_interval_us / 1_000_000.0
        return np.arange(self.n_samples, dtype=float) * dt


class SegyReader:
    """Reader capable of parsing SEG-Y revision 1 and 2 files."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)

    def read(self) -> SegyDataset:
        with self.path.open("rb") as fp:
            text_header = _decode_text_header(fp.read(_TEXT_HEADER_SIZE))
            binary_header_bytes = fp.read(_BINARY_HEADER_SIZE)
            if len(binary_header_bytes) != _BINARY_HEADER_SIZE:
                raise IOError("Truncated binary header")

            binary_header = _parse_binary_header(binary_header_bytes)
            binary_header["_raw_bytes"] = binary_header_bytes

            traces, trace_headers = _read_traces(
                fp,
                samples_per_trace=binary_header["samples_per_trace"],
                format_code=binary_header["data_sample_format"],
            )

            dataset = SegyDataset(
                text_header=text_header,
                binary_header=binary_header,
                trace_headers=trace_headers,
                data=traces,
                sample_interval_us=binary_header["sample_interval_us"],
                revision=binary_header.get("segy_revision", 1),
                path=self.path,
            )
            return dataset


def read_segy(path: os.PathLike[str] | str) -> SegyDataset:
    """Convenience wrapper returning :class:`SegyDataset`."""

    return SegyReader(path).read()


def _decode_text_header(data: bytes) -> str:
    if len(data) != _TEXT_HEADER_SIZE:
        raise IOError("Truncated textual header")

    try:
        return data.decode("ascii")
    except UnicodeDecodeError:
        return codecs.decode(data, "cp500", errors="replace")


def _parse_binary_header(data: bytes) -> Dict[str, int]:
    if len(data) != _BINARY_HEADER_SIZE:
        raise ValueError("Binary header has unexpected size")

    def _read(fmt: str, offset: int) -> int:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, data[offset : offset + size])[0]

    header = {
        "job_id": _read(">i", 0),
        "line_number": _read(">i", 4),
        "reel_number": _read(">i", 8),
        "sample_interval_us": _read(">H", 16),
        "samples_per_trace": _read(">H", 20),
        "data_sample_format": _read(">H", 24),
        "ensemble_fold": _read(">H", 26),
        "trace_sorting_code": _read(">H", 28),
        "vertical_sum_code": _read(">H", 30),
        "segy_revision": max(1, _read(">h", 300)),
        "fixed_length_traces": _read(">h", 302),
        "extended_textual_header_count": _read(">h", 304),
    }

    return header


def _parse_trace_header(data: bytes) -> Dict[str, int | bytes]:
    if len(data) != _TRACE_HEADER_SIZE:
        raise ValueError("Trace header has unexpected size")

    def _read(fmt: str, offset: int) -> int:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, data[offset : offset + size])[0]

    header: Dict[str, int | bytes] = {
        "trace_sequence_line": _read(">i", 0),
        "trace_sequence_file": _read(">i", 4),
        "field_record_number": _read(">i", 8),
        "trace_number_within_field_record": _read(">i", 12),
        "energy_source_point": _read(">i", 16),
        "ensemble_number": _read(">i", 20),
        "trace_number_within_ensemble": _read(">i", 24),
        "trace_identification_code": _read(">h", 28),
        "coordinate_scalar": _read(">h", 68),
        "source_x": _read(">i", 72),
        "source_y": _read(">i", 76),
        "group_x": _read(">i", 80),
        "group_y": _read(">i", 84),
        "elevation_receiver": _read(">i", 88),
        "elevation_source": _read(">i", 92),
        "surface_elevation_source": _read(">i", 96),
        "source_depth": _read(">i", 100),
        "datum_shift_receiver": _read(">i", 104),
        "datum_shift_source": _read(">i", 108),
        "water_depth_source": _read(">i", 112),
        "water_depth_group": _read(">i", 116),
        "samples_in_trace": _read(">H", 114),
        "sample_interval_us": _read(">H", 116),
        "gain_type": _read(">h", 118),
        "instrument_gain_constant": _read(">h", 120),
        "instrument_early_gain": _read(">h", 122),
        "correlated": _read(">h", 124),
        "sweep_frequency_start": _read(">h", 126),
        "sweep_frequency_end": _read(">h", 128),
        "sweep_length_ms": _read(">h", 130),
        "sweep_type": _read(">h", 132),
        "trace_measurement_system": _read(">h", 168),
        "scalar_to_be_applied": _read(">h", 170),
        "source_measurement": _read(">i", 172),
        "source_measurement_unit": _read(">h", 176),
        "samples_in_trace_override": _read(">H", 114),
        "_raw_bytes": data,
    }

    return header


def _read_traces(
    fp: io.BufferedReader,
    samples_per_trace: int,
    format_code: int,
) -> tuple[np.ndarray, List[Dict[str, int | bytes]]]:
    traces: List[np.ndarray] = []
    headers: List[Dict[str, int | bytes]] = []
    sample_size = _data_sample_size(format_code)

    while True:
        header_bytes = fp.read(_TRACE_HEADER_SIZE)
        if not header_bytes:
            break
        if len(header_bytes) != _TRACE_HEADER_SIZE:
            raise IOError("Truncated trace header")

        header = _parse_trace_header(header_bytes)
        this_trace_samples = header.get("samples_in_trace") or samples_per_trace
        bytes_needed = this_trace_samples * sample_size
        trace_bytes = fp.read(bytes_needed)
        if len(trace_bytes) != bytes_needed:
            raise IOError("Truncated trace data")

        trace = _decode_trace_samples(trace_bytes, format_code, this_trace_samples)
        traces.append(trace)
        headers.append(header)

    if not traces:
        raise IOError("No traces found in file")

    max_samples = max(trace.shape[0] for trace in traces)
    padded_traces = np.zeros((len(traces), max_samples), dtype=np.float32)
    for i, trace in enumerate(traces):
        padded_traces[i, : trace.shape[0]] = trace

    return padded_traces, headers


def _data_sample_size(format_code: int) -> int:
    sizes = {
        1: 4,
        2: 4,
        3: 2,
        4: 4,
        5: 4,
        6: 8,
        7: 3,
        8: 1,
        9: 8,
        10: 4,
        11: 2,
        12: 4,
        15: 8,
    }
    if format_code not in sizes:
        raise NotImplementedError(f"Unsupported SEG-Y format code {format_code}")
    return sizes[format_code]


def _decode_trace_samples(data: bytes, format_code: int, n_samples: int) -> np.ndarray:
    if format_code == 1:
        raw = np.frombuffer(data, dtype=">u4", count=n_samples)
        return _ibm_to_ieee(raw)
    if format_code == 2:
        return np.frombuffer(data, dtype=">i4", count=n_samples).astype(np.float32)
    if format_code == 3:
        return np.frombuffer(data, dtype=">i2", count=n_samples).astype(np.float32)
    if format_code == 5:
        return np.frombuffer(data, dtype=">f4", count=n_samples)
    if format_code == 6 or format_code == 9:
        return np.frombuffer(data, dtype=">f8", count=n_samples).astype(np.float32)
    if format_code == 8:
        return np.frombuffer(data, dtype=">i1", count=n_samples).astype(np.float32)
    if format_code == 4:
        return np.frombuffer(data, dtype=">i4", count=n_samples).astype(np.float32)
    if format_code in {7, 10, 11, 12, 15}:
        raise NotImplementedError(
            f"SEG-Y data sample format code {format_code} is not yet supported"
        )
    raise NotImplementedError(f"Unsupported SEG-Y format code {format_code}")


def _ibm_to_ieee(raw: np.ndarray) -> np.ndarray:
    """Vectorised conversion from IBM 32 bit floats to IEEE floats."""

    if raw.dtype != np.uint32:
        raw = raw.astype(np.uint32)

    sign = ((raw >> 31) & 0x01).astype(np.float32)
    exponent = ((raw >> 24) & 0x7F).astype(np.int32) - 64
    mantissa = (raw & 0x00FFFFFF).astype(np.float64) / float(0x01000000)

    value = np.ldexp(mantissa, exponent * 4)
    value[sign == 1] *= -1.0
    value[raw == 0] = 0.0
    return value.astype(np.float32)
