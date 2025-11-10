"""
SEG-Y reader utilities from segy_viz - robust and clean implementation
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# Mapping of SEG-Y format codes to byte widths
SAMPLE_BYTE_SIZES: Dict[int, int] = {
    1: 4,  # 4-byte IBM floating point
    2: 4,  # 4-byte signed integer
    3: 2,  # 2-byte signed integer
    5: 4,  # 4-byte IEEE floating point
    8: 1,  # 1-byte unsigned integer
}


@dataclass
class SegyMetadata:
    """Container for file-level metadata"""

    sample_interval_us: int
    samples_per_trace: int
    sample_format: int
    measurement_system: Optional[int]
    trace_count: int
    text_header: str
    binary_header: Dict[str, int]


@dataclass
class TraceHeader:
    """Minimal trace header subset for display"""

    number: int
    samples_per_trace: int
    sample_interval_us: int
    raw: bytes


@dataclass
class SegyData:
    """Holds the fully decoded SEG-Y volume"""

    samples: np.ndarray  # Shape: (sample_count, trace_count)
    metadata: SegyMetadata
    trace_headers: List[TraceHeader]

    @property
    def stats(self) -> Dict[str, float]:
        """Basic amplitude information"""
        return {
            "min": float(np.nanmin(self.samples)),
            "max": float(np.nanmax(self.samples)),
            "mean": float(np.nanmean(self.samples)),
            "std": float(np.nanstd(self.samples)),
        }


class SegyLoader:
    """Parses a SEG-Y file into numpy arrays ready for visualization"""

    def __init__(self, max_traces: Optional[int] = None) -> None:
        """
        Args:
            max_traces: Optional ceiling on the number of traces loaded
        """
        self.max_traces = max_traces
        self.data: Optional[SegyData] = None

    def load(self, path: str) -> bool:
        """
        Load SEG-Y data from path

        Returns:
            True if successful, False otherwise
        """
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

            print(f"[SEG-Y Loader] Successfully loaded: {path}")
            print(f"  Traces: {metadata.trace_count}")
            print(f"  Samples per trace: {metadata.samples_per_trace}")
            print(f"  Sample interval: {metadata.sample_interval_us} μs ({metadata.sample_interval_us / 1000:.2f} ms)")
            print(f"  Sample format: {format_code}")
            print(f"  Data range: [{self.data.stats['min']:.2e}, {self.data.stats['max']:.2e}]")
            print(f"  Mean: {self.data.stats['mean']:.2e}, Std: {self.data.stats['std']:.2e}")

            return True

        except Exception as e:
            print(f"[SEG-Y Loader] Error loading file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _decode_text_header(self, payload: bytes) -> str:
        """Convert the 3200-byte textual header to readable ASCII"""
        for encoding in ("cp500", "cp1147", "ascii"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("latin-1", errors="replace")

    def _parse_binary_header(self, payload: bytes) -> Dict[str, int]:
        """Extract a subset of commonly used binary header fields"""
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
            "measurement_system": (54, ">H"),  # 1 = meters, 2 = feet
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
    ) -> tuple:
        """Read the trace headers and amplitude samples"""
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

            trace_samples = self._decode_samples(
                amplitude_bytes, samples_per_trace, format_code
            )
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

        data = np.stack(traces, axis=1)  # shape: (samples, traces)
        return data.astype(np.float32, copy=False), trace_headers

    def _decode_samples(self, payload: bytes, sample_count: int, format_code: int) -> np.ndarray:
        """Decode amplitude data for the provided format code"""
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
        """Convert IBM 4-byte floats to IEEE 754 floats"""
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
        """Read a big-endian unsigned integer from bytes"""
        if not payload:
            return 0
        fmt = {1: ">B", 2: ">H", 4: ">I"}.get(len(payload))
        if fmt is None:
            raise ValueError("Unsupported integer size for header decoding.")
        return struct.unpack(fmt, payload)[0]

    # Compatibility methods for existing interface
    def get_data(self) -> Optional[np.ndarray]:
        """Get data array (num_samples x num_traces)"""
        if self.data is None:
            return None
        return self.data.samples

    def get_dimensions(self) -> tuple:
        """Get data dimensions (num_samples, num_traces)"""
        if self.data is None:
            return (0, 0)
        return (self.data.metadata.samples_per_trace, self.data.metadata.trace_count)

    def get_sample_rate(self) -> float:
        """Get sample rate in seconds"""
        if self.data is None:
            return 0.0
        return self.data.metadata.sample_interval_us / 1_000_000.0

    def get_text_header(self) -> str:
        """Get text header"""
        if self.data is None:
            return ""
        return self.data.metadata.text_header

    def get_binary_header(self) -> dict:
        """Get binary header dictionary"""
        if self.data is None:
            return {}
        return self.data.metadata.binary_header

    def get_trace_header(self, trace_idx: int) -> dict:
        """Get trace header for specific trace"""
        if self.data is None or trace_idx >= len(self.data.trace_headers):
            return {}

        trace_header = self.data.trace_headers[trace_idx]

        # Parse more fields from raw bytes
        raw = trace_header.raw
        return {
            'trace_number': trace_header.number,
            'samples_per_trace': trace_header.samples_per_trace,
            'sample_interval_us': trace_header.sample_interval_us,
            'trace_sequence_line': struct.unpack('>i', raw[0:4])[0],
            'trace_sequence_file': struct.unpack('>i', raw[4:8])[0],
            'field_record': struct.unpack('>i', raw[8:12])[0],
            'source_x': struct.unpack('>i', raw[72:76])[0],
            'source_y': struct.unpack('>i', raw[76:80])[0],
            'receiver_x': struct.unpack('>i', raw[80:84])[0],
            'receiver_y': struct.unpack('>i', raw[84:88])[0],
        }

    def close(self):
        """Close resources (compatibility method)"""
        self.data = None
