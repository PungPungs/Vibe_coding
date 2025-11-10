"""
SEG-Y file loader using mmap and numpy
"""
import mmap
import struct
import numpy as np


class SegyLoader:
    """mmap을 이용한 SEG-Y 파일 로더"""

    def __init__(self):
        self.file = None
        self.mmap_obj = None
        self.data = None
        self.num_traces = 0
        self.num_samples = 0
        self.sample_rate = 0.0  # 초 단위
        self.data_format = 1  # 1=IBM float, 5=IEEE float

    def load(self, filename: str) -> bool:
        """
        SEG-Y 파일을 로드합니다.

        Args:
            filename: SEG-Y 파일 경로

        Returns:
            성공 여부
        """
        try:
            # 파일 열기
            self.file = open(filename, 'rb')
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

            # Binary File Header 읽기 (3200 bytes 이후)
            self._read_binary_header()

            # Trace 개수 계산
            file_size = len(self.mmap_obj)
            trace_size = 240 + self.num_samples * 4  # trace header + data
            self.num_traces = (file_size - 3600) // trace_size

            print(f"SEG-Y Info:")
            print(f"  Traces: {self.num_traces}")
            print(f"  Samples per trace: {self.num_samples}")
            print(f"  Sample rate: {self.sample_rate * 1000:.2f} ms")
            print(f"  Data format: {self.data_format} ({'IBM' if self.data_format == 1 else 'IEEE'} float)")

            # 데이터 읽기 (정규화 없이)
            self._read_traces()

            return True

        except Exception as e:
            print(f"Error loading SEG-Y file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _read_binary_header(self):
        """Binary File Header 읽기"""
        # Binary header는 3200 bytes offset부터 시작
        offset = 3200

        # Samples per trace (bytes 3217-3218, offset 3216)
        self.num_samples = struct.unpack('>H', self.mmap_obj[3216:3218])[0]

        # Sample interval in microseconds (bytes 3221-3222, offset 3220)
        sample_interval_us = struct.unpack('>H', self.mmap_obj[3220:3222])[0]
        self.sample_rate = sample_interval_us / 1_000_000.0  # 초 단위로 변환

        # Data sample format code (bytes 3225-3226, offset 3224)
        self.data_format = struct.unpack('>H', self.mmap_obj[3224:3226])[0]

    def _read_traces(self):
        """모든 trace 데이터 읽기"""
        # 데이터 배열 초기화 (num_samples x num_traces)
        self.data = np.zeros((self.num_samples, self.num_traces), dtype=np.float32)

        # 각 trace 읽기
        offset = 3600  # Text header (3200) + Binary header (400)

        for trace_idx in range(self.num_traces):
            # Trace header 건너뛰기 (240 bytes)
            data_offset = offset + 240

            # Trace data 읽기
            trace_bytes = self.mmap_obj[data_offset:data_offset + self.num_samples * 4]

            # Format에 따라 변환
            if self.data_format == 5:
                # IEEE float (big-endian)
                trace_data = np.frombuffer(trace_bytes, dtype='>f4')
            else:
                # IBM float (format code 1) - IEEE로 근사
                trace_data = self._ibm_to_ieee(trace_bytes)

            self.data[:, trace_idx] = trace_data

            # 다음 trace로 이동
            offset += 240 + self.num_samples * 4

    def _ibm_to_ieee(self, ibm_bytes: bytes) -> np.ndarray:
        """IBM float를 IEEE float로 변환 (간단한 버전)"""
        # IBM float 4바이트를 읽어서 IEEE float로 변환
        n = len(ibm_bytes) // 4
        ieee_data = np.zeros(n, dtype=np.float32)

        for i in range(n):
            ibm_int = struct.unpack('>I', ibm_bytes[i*4:(i+1)*4])[0]

            if ibm_int == 0:
                ieee_data[i] = 0.0
                continue

            # IBM float 구조: sign(1) exponent(7) mantissa(24)
            sign = (ibm_int >> 31) & 1
            exponent = (ibm_int >> 24) & 0x7F
            mantissa = ibm_int & 0x00FFFFFF

            # IBM: value = (-1)^sign * 0.mantissa * 16^(exponent - 64)
            # IEEE로 변환
            if mantissa != 0:
                # mantissa를 정규화
                while (mantissa & 0x00800000) == 0:
                    mantissa <<= 1
                    exponent -= 1

                # IEEE float 변환
                value = mantissa / (1 << 24) * (16 ** (exponent - 64))
                if sign:
                    value = -value
                ieee_data[i] = value
            else:
                ieee_data[i] = 0.0

        return ieee_data

    def get_data(self) -> np.ndarray:
        """데이터 반환 (num_samples x num_traces)"""
        return self.data

    def get_dimensions(self) -> tuple:
        """데이터 크기 반환 (num_samples, num_traces)"""
        return (self.num_samples, self.num_traces)

    def get_sample_rate(self) -> float:
        """Sample rate 반환 (초 단위)"""
        return self.sample_rate

    def close(self):
        """파일 닫기"""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file is not None:
            self.file.close()
