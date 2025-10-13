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
from typing import ClassVar, Dict, List, Optional
import codecs
import io
import os
import struct

import numpy as np

_TEXT_HEADER_SIZE = 3200
_BINARY_HEADER_SIZE = 400
_TRACE_HEADER_SIZE = 240


@dataclass(frozen=True)
class BinaryHeader:
    """Structured representation of the 400 byte binary header."""

    job_id: int
    line_number: int
    reel_number: int
    sample_interval_us: int
    samples_per_trace: int
    data_sample_format: int
    ensemble_fold: int
    trace_sorting_code: int
    vertical_sum_code: int
    segy_revision: int
    fixed_length_traces: int
    extended_textual_header_count: int
    endianness: str
    raw_bytes: bytes

    _FIELDS: ClassVar[tuple[tuple[str, str, int], ...]] = (
        ("job_id", "i", 0),
        ("line_number", "i", 4),
        ("reel_number", "i", 8),
        ("sample_interval_us", "H", 16),
        ("samples_per_trace", "H", 20),
        ("data_sample_format", "H", 24),
        ("ensemble_fold", "H", 26),
        ("trace_sorting_code", "H", 28),
        ("vertical_sum_code", "H", 30),
        ("segy_revision", "h", 300),
        ("fixed_length_traces", "h", 302),
        ("extended_textual_header_count", "h", 304),
    )

    _SUPPORTED_SAMPLE_FORMATS: ClassVar[set[int]] = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        15,
    }

    @classmethod
    def from_bytes(cls, data: bytes) -> "BinaryHeader":
        """Parse a binary header, automatically resolving endianness."""

        if len(data) != _BINARY_HEADER_SIZE:
            raise ValueError("Binary header has unexpected size")

        for endian, label in ((">", "big"), ("<", "little")):
            values: Dict[str, int] = {}
            for name, fmt, offset in cls._FIELDS:
                struct_obj = struct.Struct(endian + fmt)
                values[name] = struct_obj.unpack_from(data, offset)[0]

            values["segy_revision"] = max(1, values["segy_revision"])

            fmt_code = values["data_sample_format"]
            if (
                fmt_code in cls._SUPPORTED_SAMPLE_FORMATS
                and values["samples_per_trace"] > 0
                and values["sample_interval_us"] > 0
            ):
                return cls(raw_bytes=data, endianness=label, **values)

        raise ValueError("Unable to determine SEG-Y binary header endianness")

    def to_dict(self) -> Dict[str, int | bytes | str]:
        return {
            "job_id": self.job_id,
            "line_number": self.line_number,
            "reel_number": self.reel_number,
            "sample_interval_us": self.sample_interval_us,
            "samples_per_trace": self.samples_per_trace,
            "data_sample_format": self.data_sample_format,
            "ensemble_fold": self.ensemble_fold,
            "trace_sorting_code": self.trace_sorting_code,
            "vertical_sum_code": self.vertical_sum_code,
            "segy_revision": self.segy_revision,
            "fixed_length_traces": self.fixed_length_traces,
            "extended_textual_header_count": self.extended_textual_header_count,
            "endianness": self.endianness,
            "_raw_bytes": self.raw_bytes,
        }


@dataclass(frozen=True)
class TraceHeader:
    """Structured representation of the 240 byte trace header."""

    trace_sequence_line: int
    trace_sequence_file: int
    field_record_number: int
    trace_number_within_field_record: int
    energy_source_point: int
    ensemble_number: int
    trace_number_within_ensemble: int
    trace_identification_code: int
    coordinate_scalar: int
    source_x: int
    source_y: int
    group_x: int
    group_y: int
    samples_in_trace: int
    sample_interval_us: int
    gain_type: int
    instrument_gain_constant: int
    instrument_early_gain: int
    correlated: int
    sweep_frequency_start: int
    sweep_frequency_end: int
    sweep_length_ms: int
    sweep_type: int
    trace_measurement_system: int
    scalar_to_be_applied: int
    source_measurement: int
    source_measurement_unit: int
    raw_bytes: bytes

    _FIELDS: ClassVar[tuple[tuple[str, str, int], ...]] = (
        ("trace_sequence_line", "i", 0),
        ("trace_sequence_file", "i", 4),
        ("field_record_number", "i", 8),
        ("trace_number_within_field_record", "i", 12),
        ("energy_source_point", "i", 16),
        ("ensemble_number", "i", 20),
        ("trace_number_within_ensemble", "i", 24),
        ("trace_identification_code", "h", 28),
        ("coordinate_scalar", "h", 68),
        ("source_x", "i", 72),
        ("source_y", "i", 76),
        ("group_x", "i", 80),
        ("group_y", "i", 84),
        ("samples_in_trace", "H", 114),
        ("sample_interval_us", "H", 116),
        ("gain_type", "h", 118),
        ("instrument_gain_constant", "h", 120),
        ("instrument_early_gain", "h", 122),
        ("correlated", "h", 124),
        ("sweep_frequency_start", "h", 126),
        ("sweep_frequency_end", "h", 128),
        ("sweep_length_ms", "h", 130),
        ("sweep_type", "h", 132),
        ("trace_measurement_system", "h", 168),
        ("scalar_to_be_applied", "h", 170),
        ("source_measurement", "i", 172),
        ("source_measurement_unit", "h", 176),
    )

    @classmethod
    def from_bytes(cls, data: bytes, *, endianness: str) -> "TraceHeader":
        if len(data) != _TRACE_HEADER_SIZE:
            raise ValueError("Trace header has unexpected size")

        endian = ">" if endianness == "big" else "<"
        values: Dict[str, int] = {}
        for name, fmt, offset in cls._FIELDS:
            struct_obj = struct.Struct(endian + fmt)
            values[name] = struct_obj.unpack_from(data, offset)[0]

        return cls(raw_bytes=data, **values)

    def to_dict(self) -> Dict[str, int | bytes]:
        result: Dict[str, int | bytes] = {
            "trace_sequence_line": self.trace_sequence_line,
            "trace_sequence_file": self.trace_sequence_file,
            "field_record_number": self.field_record_number,
            "trace_number_within_field_record": self.trace_number_within_field_record,
            "energy_source_point": self.energy_source_point,
            "ensemble_number": self.ensemble_number,
            "trace_number_within_ensemble": self.trace_number_within_ensemble,
            "trace_identification_code": self.trace_identification_code,
            "coordinate_scalar": self.coordinate_scalar,
            "source_x": self.source_x,
            "source_y": self.source_y,
            "group_x": self.group_x,
            "group_y": self.group_y,
            "samples_in_trace": self.samples_in_trace,
            "sample_interval_us": self.sample_interval_us,
            "gain_type": self.gain_type,
            "instrument_gain_constant": self.instrument_gain_constant,
            "instrument_early_gain": self.instrument_early_gain,
            "correlated": self.correlated,
            "sweep_frequency_start": self.sweep_frequency_start,
            "sweep_frequency_end": self.sweep_frequency_end,
            "sweep_length_ms": self.sweep_length_ms,
            "sweep_type": self.sweep_type,
            "trace_measurement_system": self.trace_measurement_system,
            "scalar_to_be_applied": self.scalar_to_be_applied,
            "source_measurement": self.source_measurement,
            "source_measurement_unit": self.source_measurement_unit,
            "_raw_bytes": self.raw_bytes,
        }
        return result


@dataclass(frozen=True)
class SegyDataset:
    """Container object storing SEG-Y traces and metadata."""

    text_header: str
    binary_header: BinaryHeader
    trace_headers: List[TraceHeader]
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

    def binary_header_dict(self) -> Dict[str, int | bytes | str]:
        """Return the binary header as a dictionary."""

        return self.binary_header.to_dict()

    def trace_header_dicts(self) -> List[Dict[str, int | bytes]]:
        """Return a list with all trace headers converted to dictionaries."""

        return [header.to_dict() for header in self.trace_headers]


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

            binary_header = BinaryHeader.from_bytes(binary_header_bytes)

            traces, trace_headers = _read_traces(
                fp,
                samples_per_trace=binary_header.samples_per_trace,
                format_code=binary_header.data_sample_format,
                endianness=binary_header.endianness,
            )

            dataset = SegyDataset(
                text_header=text_header,
                binary_header=binary_header,
                trace_headers=trace_headers,
                data=traces,
                sample_interval_us=binary_header.sample_interval_us,
                revision=binary_header.segy_revision,
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


def _read_traces(
    fp: io.BufferedReader,
    samples_per_trace: int,
    format_code: int,
    *,
    endianness: str,
) -> tuple[np.ndarray, List[TraceHeader]]:
    traces: List[np.ndarray] = []
    headers: List[TraceHeader] = []
    sample_size = _data_sample_size(format_code)

    while True:
        header_bytes = fp.read(_TRACE_HEADER_SIZE)
        if not header_bytes:
            break
        if len(header_bytes) != _TRACE_HEADER_SIZE:
            raise IOError("Truncated trace header")

        header = TraceHeader.from_bytes(header_bytes, endianness=endianness)
        this_trace_samples = header.samples_in_trace or samples_per_trace
        bytes_needed = this_trace_samples * sample_size
        trace_bytes = fp.read(bytes_needed)
        if len(trace_bytes) != bytes_needed:
            raise IOError("Truncated trace data")

        trace = _decode_trace_samples(
            trace_bytes, format_code, this_trace_samples, endianness=endianness
        )
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


def _decode_trace_samples(
    data: bytes, format_code: int, n_samples: int, *, endianness: str
) -> np.ndarray:
    dtype_prefix = ">" if endianness == "big" else "<"
    if format_code == 1:
        raw = np.frombuffer(data, dtype=f"{dtype_prefix}u4", count=n_samples)
        return _ibm_to_ieee(raw)
    if format_code == 2:
        return np.frombuffer(data, dtype=f"{dtype_prefix}i4", count=n_samples).astype(
            np.float32
        )
    if format_code == 3:
        return np.frombuffer(data, dtype=f"{dtype_prefix}i2", count=n_samples).astype(
            np.float32
        )
    if format_code == 5:
        return np.frombuffer(data, dtype=f"{dtype_prefix}f4", count=n_samples)
    if format_code == 6 or format_code == 9:
        return np.frombuffer(data, dtype=f"{dtype_prefix}f8", count=n_samples).astype(
            np.float32
        )
    if format_code == 8:
        return np.frombuffer(data, dtype=f"{dtype_prefix}i1", count=n_samples).astype(
            np.float32
        )
    if format_code == 4:
        return np.frombuffer(data, dtype=f"{dtype_prefix}i4", count=n_samples).astype(
            np.float32
        )
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
