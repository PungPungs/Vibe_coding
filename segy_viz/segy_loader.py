"""SEG-Y reader utilities."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

SAMPLE_BYTE_SIZES: Dict[int, int] = {
    1: 4,
    2: 4,
    3: 2,
    5: 4,
    8: 1,
}


@dataclass
class SegyMetadata:
    sample_interval_us: int
    samples_per_trace: int
    sample_format: int
    measurement_system: Optional[int]
    trace_count: int
    text_header: str
    binary_header: Dict[str, int]


@dataclass
class TraceHeader:
    number: int
    samples_per_trace: int
    sample_interval_us: int
    raw: bytes


@dataclass
class SegyData:
    samples: np.ndarray
    metadata: SegyMetadata
    trace_headers: List[TraceHeader]

    @property
    def stats(self) -> Dict[str, float]:
        return {
            "min": float(np.nanmin(self.samples)),
            "max": float(np.nanmax(self.samples)),
            "mean": float(np.nanmean(self.samples)),
            "std": float(np.nanstd(self.samples)),
        }


class SegyLoader:
    def __init__(self, max_traces: Optional[int] = None) -> None:
        self.max_traces = max_traces
        self.data: Optional[SegyData] = None

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as fp:
                text_header = self._decode_text_header(fp.read(3200))
                binary_header_bytes = fp.read(400)
                if len(binary_header_bytes) != 400:
                    raise IOError("Binary header truncated – not a valid SEG-Y file.")
                binary_header = self._parse_binary_header(binary_header_bytes)
                format_code = binary_header.get("sample_format", 5)
                if format_code not in SAMPLE_BYTE_SIZES:
                    raise ValueError(f"Unsupported SEG-Y sample format code: {format_code}")
                traces, trace_headers = self._read_traces(
                    fp,
                    format_code=format_code,
                    default_samples=binary_header.get("samples_per_trace", 0),
                    default_interval=binary_header.get("sample_interval_us", 0),
                )

            metadata = SegyMetadata(
                sample_interval_us=binary_header.get("sample_interval_us", 0),
                samples_per_trace=traces.shape[0] if traces.size else 0,
                sample_format=format_code,
                measurement_system=binary_header.get("measurement_system"),
                trace_count=traces.shape[1],
                text_header=text_header,
                binary_header=binary_header,
            )
            self.data = SegyData(samples=traces, metadata=metadata, trace_headers=trace_headers)
            return True
        except Exception as exc:  # pragma: no cover - IO errors
            print(f"[SEG-Y Loader] Error: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def get_data(self) -> np.ndarray:
        if not self.data:
            raise RuntimeError("No SEG-Y data loaded")
        return self.data.samples

    def get_dimensions(self) -> tuple[int, int]:
        if not self.data:
            return 0, 0
        return self.data.samples.shape

    def get_sample_rate(self) -> float:
        if not self.data:
            return 0.0
        return self.data.metadata.sample_interval_us / 1_000_000.0

    def get_header_summary(self) -> str:
        if not self.data:
            return "No data loaded"
        info = ["SEG-Y Header Summary", "---------------------"]
        info.append(self.data.metadata.text_header)
        info.append("")
        info.append("Binary Header:")
        for key, value in self.data.metadata.binary_header.items():
            info.append(f"  {key}: {value}")
        info.append("")
        info.append("Trace Statistics:")
        for key, value in self.data.stats.items():
            info.append(f"  {key}: {value:.3e}")
        return "\n".join(info)

    def _decode_text_header(self, payload: bytes) -> str:
        for encoding in ("cp500", "cp1147", "ascii"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("latin-1", errors="replace")

    def _parse_binary_header(self, payload: bytes) -> Dict[str, int]:
        field_map = {
            "job_id": (0, ">i"),
            "line_number": (4, ">i"),
            "reel_number": (8, ">i"),
            "data_traces_per_ensemble": (12, ">H"),
            "aux_traces_per_ensemble": (14, ">H"),
            "sample_interval_us": (16, ">H"),
            "sample_interval_original_us": (18, ">H"),
            "samples_per_trace": (20, ">H"),
            "samples_per_trace_original": (22, ">H"),
            "sample_format": (24, ">H"),
            "data_sample_type": (28, ">H"),
            "measurement_system": (54, ">H"),
        }
        header: Dict[str, int] = {}
        for name, (offset, fmt) in field_map.items():
            size = struct.calcsize(fmt)
            chunk = payload[offset : offset + size]
            if len(chunk) == size:
                header[name] = struct.unpack(fmt, chunk)[0]
        return header

    def _read_traces(
        self,
        fp: io.BufferedReader,
        *,
        format_code: int,
        default_samples: int,
        default_interval: int,
    ) -> tuple[np.ndarray, List[TraceHeader]]:
        sample_width = SAMPLE_BYTE_SIZES[format_code]
        traces: List[np.ndarray] = []
        trace_headers: List[TraceHeader] = []
        trace_number = 0

        while True:
            header_bytes = fp.read(240)
            if not header_bytes:
                break
            if len(header_bytes) != 240:
                raise IOError(f"Trace header #{trace_number + 1} truncated.")

            samples_per_trace = self._read_uint(header_bytes[114:116]) or default_samples
            sample_interval_us = self._read_uint(header_bytes[116:118]) or default_interval

            if samples_per_trace == 0:
                raise ValueError(
                    f"Trace header #{trace_number + 1} missing sample count information."
                )

            amplitude_bytes = fp.read(samples_per_trace * sample_width)
            if len(amplitude_bytes) != samples_per_trace * sample_width:
                raise IOError(f"Trace #{trace_number + 1} data payload truncated.")

            trace_samples = self._decode_samples(amplitude_bytes, samples_per_trace, format_code)
            traces.append(trace_samples)
            trace_headers.append(
                TraceHeader(
                    number=trace_number + 1,
                    samples_per_trace=samples_per_trace,
                    sample_interval_us=sample_interval_us,
                    raw=header_bytes,
                )
            )

            trace_number += 1
            if self.max_traces and trace_number >= self.max_traces:
                break

        if not traces:
            raise ValueError("SEG-Y file did not contain any complete traces.")

        data = np.stack(traces, axis=1)
        return data.astype(np.float32, copy=False), trace_headers

    def _decode_samples(self, payload: bytes, sample_count: int, format_code: int) -> np.ndarray:
        if format_code == 1:
            return self._decode_ibm_floats(payload, sample_count)
        if format_code == 2:
            return np.frombuffer(payload, dtype=">i4", count=sample_count).astype(np.float32)
        if format_code == 3:
            return np.frombuffer(payload, dtype=">i2", count=sample_count).astype(np.float32)
        if format_code == 5:
            return np.frombuffer(payload, dtype=">f4", count=sample_count).astype(np.float32)
        if format_code == 8:
            return np.frombuffer(payload, dtype=">u1", count=sample_count).astype(np.float32)
        raise ValueError(f"Unsupported sample format code: {format_code}")

    def _decode_ibm_floats(self, payload: bytes, sample_count: int) -> np.ndarray:
        raw = np.frombuffer(payload, dtype=">u4", count=sample_count)
        if raw.size == 0:
            return raw.astype(np.float32)
        sign = ((raw >> 31) & 0x01).astype(np.int8)
        exponent = ((raw >> 24) & 0x7F).astype(np.int16) - 64
        fraction = (raw & 0x00FFFFFF).astype(np.float64) / float(0x01000000)
        magnitude = np.power(16.0, exponent, dtype=np.float64) * fraction
        magnitude[raw == 0] = 0.0
        return (np.where(sign == 0, 1.0, -1.0) * magnitude).astype(np.float32)

    def _read_uint(self, payload: bytes) -> int:
        if not payload:
            return 0
        fmt = {1: ">B", 2: ">H", 4: ">I"}.get(len(payload))
        if fmt is None:
            raise ValueError("Unsupported integer size for header decoding.")
        return struct.unpack(fmt, payload)[0]
