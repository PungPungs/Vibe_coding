"""
Create a simple test SEG-Y file for testing
"""
import numpy as np
import struct


def create_test_segy(filename: str, num_traces: int = 50, num_samples: int = 500):
    """Create a simple test SEG-Y file with synthetic data"""

    with open(filename, 'wb') as f:
        # Text header (3200 bytes) - EBCDIC
        text_header = b'C 1 TEST SEG-Y FILE' + b' ' * (3200 - 19)
        f.write(text_header)

        # Binary header (400 bytes)
        binary_header = bytearray(400)

        # Samples per trace (bytes 21-22, 0-indexed: 20-21)
        struct.pack_into('>H', binary_header, 20, num_samples)

        # Sample interval in microseconds (bytes 17-18, 0-indexed: 16-17)
        # 2000 us = 2 ms
        struct.pack_into('>H', binary_header, 16, 2000)

        # Data format code (bytes 25-26, 0-indexed: 24-25)
        # 5 = IEEE floating point
        struct.pack_into('>H', binary_header, 24, 5)

        f.write(binary_header)

        # Create synthetic seismic data
        time = np.arange(num_samples)

        for trace_idx in range(num_traces):
            # Trace header (240 bytes)
            trace_header = bytearray(240)

            # Trace sequence number
            struct.pack_into('>i', trace_header, 0, trace_idx + 1)
            struct.pack_into('>i', trace_header, 4, trace_idx + 1)

            # Number of samples
            struct.pack_into('>H', trace_header, 114, num_samples)

            # Sample interval
            struct.pack_into('>H', trace_header, 116, 2000)

            f.write(trace_header)

            # Generate synthetic seismic trace
            # Ricker wavelet with varying arrival time
            arrival_time = 100 + trace_idx * 2  # Linear moveout
            freq = 25.0  # Hz

            # Ricker wavelet
            t_shifted = (time - arrival_time) / 1000.0
            ricker = (1.0 - 2.0 * (np.pi * freq * t_shifted)**2) * \
                     np.exp(-(np.pi * freq * t_shifted)**2)

            # Add some noise
            noise = np.random.normal(0, 0.1, num_samples)
            trace_data = ricker + noise

            # Add another event
            arrival_time2 = 300 + trace_idx * 1.5
            t_shifted2 = (time - arrival_time2) / 1000.0
            ricker2 = 0.7 * (1.0 - 2.0 * (np.pi * freq * t_shifted2)**2) * \
                      np.exp(-(np.pi * freq * t_shifted2)**2)
            trace_data += ricker2

            # Scale to reasonable amplitude
            trace_data = trace_data * 1000.0

            # Write as IEEE float (big-endian)
            trace_bytes = trace_data.astype('>f4').tobytes()
            f.write(trace_bytes)

    print(f"Created test SEG-Y file: {filename}")
    print(f"  Traces: {num_traces}")
    print(f"  Samples: {num_samples}")
    print(f"  Sample interval: 2 ms")


if __name__ == '__main__':
    create_test_segy('test_data.sgy', num_traces=100, num_samples=600)
