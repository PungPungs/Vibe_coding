"""
Enhanced SEG-Y file loader with header information
"""
import mmap
import struct
import numpy as np


class SegyLoader:
    """Enhanced SEG-Y loader with header parsing"""

    def __init__(self):
        self.file = None
        self.mmap_obj = None
        self.data = None
        self.num_traces = 0
        self.num_samples = 0
        self.sample_rate = 0.0
        self.data_format = 1

        # Headers
        self.text_header = ""
        self.binary_header = {}
        self.trace_headers = []

    def load(self, filename: str) -> bool:
        """Load SEG-Y file with header information"""
        try:
            self.file = open(filename, 'rb')
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

            # Read headers
            self._read_text_header()
            self._read_binary_header()

            # Calculate trace count
            file_size = len(self.mmap_obj)
            trace_size = 240 + self.num_samples * 4
            self.num_traces = (file_size - 3600) // trace_size

            print(f"[SEG-Y Loader] File loaded successfully")
            print(f"  Traces: {self.num_traces}")
            print(f"  Samples per trace: {self.num_samples}")
            print(f"  Sample rate: {self.sample_rate * 1000:.2f} ms")
            print(f"  Data format: {self.data_format}")

            # Read trace data and headers
            self._read_traces()

            return True

        except Exception as e:
            print(f"[SEG-Y Loader] Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _read_text_header(self):
        """Read 3200-byte text header"""
        text_bytes = self.mmap_obj[0:3200]
        try:
            # Try EBCDIC decoding
            self.text_header = self._ebcdic_to_ascii(text_bytes)
        except:
            # Fallback to ASCII
            self.text_header = text_bytes.decode('ascii', errors='ignore')

    def _ebcdic_to_ascii(self, ebcdic_bytes: bytes) -> str:
        """Convert EBCDIC to ASCII"""
        # Simple EBCDIC to ASCII conversion table
        ebcdic_to_ascii_table = bytes.maketrans(
            bytes(range(256)),
            bytes([
                0x00, 0x01, 0x02, 0x03, 0x9C, 0x09, 0x86, 0x7F,
                0x97, 0x8D, 0x8E, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                0x10, 0x11, 0x12, 0x13, 0x9D, 0x85, 0x08, 0x87,
                0x18, 0x19, 0x92, 0x8F, 0x1C, 0x1D, 0x1E, 0x1F,
                0x80, 0x81, 0x82, 0x83, 0x84, 0x0A, 0x17, 0x1B,
                0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x05, 0x06, 0x07,
                0x90, 0x91, 0x16, 0x93, 0x94, 0x95, 0x96, 0x04,
                0x98, 0x99, 0x9A, 0x9B, 0x14, 0x15, 0x9E, 0x1A,
                0x20, 0xA0, 0xE2, 0xE4, 0xE0, 0xE1, 0xE3, 0xE5,
                0xE7, 0xF1, 0xA2, 0x2E, 0x3C, 0x28, 0x2B, 0x7C,
                0x26, 0xE9, 0xEA, 0xEB, 0xE8, 0xED, 0xEE, 0xEF,
                0xEC, 0xDF, 0x21, 0x24, 0x2A, 0x29, 0x3B, 0xAC,
                0x2D, 0x2F, 0xC2, 0xC4, 0xC0, 0xC1, 0xC3, 0xC5,
                0xC7, 0xD1, 0xA6, 0x2C, 0x25, 0x5F, 0x3E, 0x3F,
                0xF8, 0xC9, 0xCA, 0xCB, 0xC8, 0xCD, 0xCE, 0xCF,
                0xCC, 0x60, 0x3A, 0x23, 0x40, 0x27, 0x3D, 0x22,
                0xD8, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
                0x68, 0x69, 0xAB, 0xBB, 0xF0, 0xFD, 0xFE, 0xB1,
                0xB0, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,
                0x71, 0x72, 0xAA, 0xBA, 0xE6, 0xB8, 0xC6, 0xA4,
                0xB5, 0x7E, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
                0x79, 0x7A, 0xA1, 0xBF, 0xD0, 0xDD, 0xDE, 0xAE,
                0x5E, 0xA3, 0xA5, 0xB7, 0xA9, 0xA7, 0xB6, 0xBC,
                0xBD, 0xBE, 0x5B, 0x5D, 0xAF, 0xA8, 0xB4, 0xD7,
                0x7B, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
                0x48, 0x49, 0xAD, 0xF4, 0xF6, 0xF2, 0xF3, 0xF5,
                0x7D, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,
                0x51, 0x52, 0xB9, 0xFB, 0xFC, 0xF9, 0xFA, 0xFF,
                0x5C, 0xF7, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                0x59, 0x5A, 0xB2, 0xD4, 0xD6, 0xD2, 0xD3, 0xD5,
                0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                0x38, 0x39, 0xB3, 0xDB, 0xDC, 0xD9, 0xDA, 0x9F
            ])
        )
        return ebcdic_bytes.translate(ebcdic_to_ascii_table).decode('ascii', errors='ignore')

    def _read_binary_header(self):
        """Read binary file header"""
        offset = 3200

        # Parse key fields
        self.num_samples = struct.unpack('>H', self.mmap_obj[3216:3218])[0]
        sample_interval_us = struct.unpack('>H', self.mmap_obj[3220:3222])[0]
        self.sample_rate = sample_interval_us / 1_000_000.0
        self.data_format = struct.unpack('>H', self.mmap_obj[3224:3226])[0]

        # Store full binary header info
        self.binary_header = {
            'job_id': struct.unpack('>i', self.mmap_obj[3200:3204])[0],
            'line_number': struct.unpack('>i', self.mmap_obj[3204:3208])[0],
            'reel_number': struct.unpack('>i', self.mmap_obj[3208:3212])[0],
            'traces_per_ensemble': struct.unpack('>H', self.mmap_obj[3212:3214])[0],
            'auxiliary_traces': struct.unpack('>H', self.mmap_obj[3214:3216])[0],
            'samples_per_trace': self.num_samples,
            'sample_interval_us': sample_interval_us,
            'sample_interval_original': struct.unpack('>H', self.mmap_obj[3222:3224])[0],
            'data_format': self.data_format,
            'ensemble_fold': struct.unpack('>H', self.mmap_obj[3226:3228])[0],
            'trace_sorting': struct.unpack('>H', self.mmap_obj[3228:3230])[0],
        }

    def _read_traces(self):
        """Read all trace data and headers"""
        self.data = np.zeros((self.num_samples, self.num_traces), dtype=np.float32)
        self.trace_headers = []

        offset = 3600

        for trace_idx in range(self.num_traces):
            # Read trace header (240 bytes)
            header = self._read_trace_header(offset)
            self.trace_headers.append(header)

            # Read trace data
            data_offset = offset + 240
            trace_bytes = self.mmap_obj[data_offset:data_offset + self.num_samples * 4]

            if self.data_format == 5:
                trace_data = np.frombuffer(trace_bytes, dtype='>f4')
            else:
                trace_data = self._ibm_to_ieee(trace_bytes)

            self.data[:, trace_idx] = trace_data

            offset += 240 + self.num_samples * 4

    def _read_trace_header(self, offset: int) -> dict:
        """Read trace header (simplified)"""
        header = {
            'trace_sequence_line': struct.unpack('>i', self.mmap_obj[offset:offset+4])[0],
            'trace_sequence_file': struct.unpack('>i', self.mmap_obj[offset+4:offset+8])[0],
            'field_record': struct.unpack('>i', self.mmap_obj[offset+8:offset+12])[0],
            'trace_number': struct.unpack('>i', self.mmap_obj[offset+12:offset+16])[0],
            'source_x': struct.unpack('>i', self.mmap_obj[offset+72:offset+76])[0],
            'source_y': struct.unpack('>i', self.mmap_obj[offset+76:offset+80])[0],
            'receiver_x': struct.unpack('>i', self.mmap_obj[offset+80:offset+84])[0],
            'receiver_y': struct.unpack('>i', self.mmap_obj[offset+84:offset+88])[0],
            'num_samples': struct.unpack('>H', self.mmap_obj[offset+114:offset+116])[0],
            'sample_interval': struct.unpack('>H', self.mmap_obj[offset+116:offset+118])[0],
        }
        return header

    def _ibm_to_ieee(self, ibm_bytes: bytes) -> np.ndarray:
        """Convert IBM float to IEEE float"""
        n = len(ibm_bytes) // 4
        ieee_data = np.zeros(n, dtype=np.float32)

        for i in range(n):
            ibm_int = struct.unpack('>I', ibm_bytes[i*4:(i+1)*4])[0]

            if ibm_int == 0:
                ieee_data[i] = 0.0
                continue

            sign = (ibm_int >> 31) & 1
            exponent = (ibm_int >> 24) & 0x7F
            mantissa = ibm_int & 0x00FFFFFF

            if mantissa != 0:
                while (mantissa & 0x00800000) == 0:
                    mantissa <<= 1
                    exponent -= 1

                value = mantissa / (1 << 24) * (16 ** (exponent - 64))
                if sign:
                    value = -value
                ieee_data[i] = value
            else:
                ieee_data[i] = 0.0

        return ieee_data

    def get_data(self) -> np.ndarray:
        """Get data array (num_samples x num_traces)"""
        return self.data

    def get_dimensions(self) -> tuple:
        """Get data dimensions (num_samples, num_traces)"""
        return (self.num_samples, self.num_traces)

    def get_sample_rate(self) -> float:
        """Get sample rate in seconds"""
        return self.sample_rate

    def get_text_header(self) -> str:
        """Get text header"""
        return self.text_header

    def get_binary_header(self) -> dict:
        """Get binary header dictionary"""
        return self.binary_header

    def get_trace_header(self, trace_idx: int) -> dict:
        """Get trace header for specific trace"""
        if 0 <= trace_idx < len(self.trace_headers):
            return self.trace_headers[trace_idx]
        return {}

    def close(self):
        """Close file"""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file is not None:
            self.file.close()
