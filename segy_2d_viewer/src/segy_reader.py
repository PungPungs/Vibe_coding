"""
SEG-Y 파일 읽기 및 파싱 모듈 (mmap + numpy 최적화)
"""
import numpy as np
import struct
import mmap
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
import os


class SegyReader:
    """SEG-Y 파일을 읽고 데이터를 처리하는 클래스 (mmap + numpy 기반)"""

    def __init__(self):
        self.filename: Optional[str] = None
        self.data: Optional[np.ndarray] = None
        self.raw_data: Optional[np.ndarray] = None  # 정규화 전 원본 데이터
        self.num_traces: int = 0
        self.num_samples: int = 0
        self.sample_rate: float = 0.0
        self.mmap_obj: Optional[mmap.mmap] = None
        self.file_obj = None

    def load_file(self, filename: str, use_parallel: bool = True) -> bool:
        """
        SEG-Y 파일을 로드합니다 (mmap + numpy 사용).

        Args:
            filename: SEG-Y 파일 경로
            use_parallel: 병렬 처리 사용 여부

        Returns:
            성공 여부
        """
        try:
            self.filename = filename

            # 파일 열기
            self.file_obj = open(filename, 'rb')
            self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)

            # 바이너리 파일 헤더 읽기 (3200 bytes 텍스트 + 400 bytes 바이너리)
            binary_header_offset = 3200

            # 샘플 수 읽기 (byte 3220-3221, 2 bytes)
            self.num_samples = struct.unpack('>H', self.mmap_obj[3220:3222])[0]

            # 샘플링 간격 읽기 (byte 3216-3217, 2 bytes, microseconds)
            sample_interval_us = struct.unpack('>H', self.mmap_obj[3216:3218])[0]
            self.sample_rate = sample_interval_us / 1_000_000.0  # seconds

            # 파일 크기로부터 트레이스 수 계산
            file_size = os.path.getsize(filename)
            header_size = 3600  # 3200 + 400
            trace_size = 240 + self.num_samples * 4  # 240 byte header + samples (4 bytes each)
            self.num_traces = (file_size - header_size) // trace_size

            print(f"Loading SEG-Y: {self.num_traces} traces, {self.num_samples} samples")

            # 데이터 읽기 (병렬 또는 순차)
            if use_parallel and self.num_traces > 100:
                self._load_data_parallel()
            else:
                self._load_data_sequential()

            # 원본 데이터 저장
            self.raw_data = self.data.copy()

            # 데이터 정규화 (시각화를 위해)
            self._normalize_data()

            return True

        except Exception as e:
            print(f"Error loading SEG-Y file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_data_sequential(self):
        """순차적으로 데이터를 로드합니다."""
        self.data = np.zeros((self.num_samples, self.num_traces), dtype=np.float32)

        header_size = 3600
        trace_size = 240 + self.num_samples * 4

        for i in range(self.num_traces):
            trace_offset = header_size + i * trace_size + 240  # 트레이스 헤더 건너뛰기

            # IBM float를 읽어서 IEEE float로 변환
            trace_bytes = self.mmap_obj[trace_offset:trace_offset + self.num_samples * 4]
            trace_data = self._ibm_to_float(trace_bytes, self.num_samples)

            # 가로로 저장 (transpose)
            self.data[:, i] = trace_data

    def _load_data_parallel(self):
        """병렬로 데이터를 로드합니다."""
        header_size = 3600
        trace_size = 240 + self.num_samples * 4

        # 트레이스 인덱스 리스트 생성
        trace_indices = list(range(self.num_traces))

        # CPU 코어 수
        num_processes = min(cpu_count(), 8)  # 최대 8개 프로세스

        # 청크 크기 계산
        chunk_size = max(1, self.num_traces // (num_processes * 4))

        # 작업 리스트 생성
        tasks = [(self.filename, i, self.num_samples, header_size, trace_size)
                 for i in trace_indices]

        # 병렬 처리
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(_load_trace_worker, tasks, chunksize=chunk_size)

        # 결과 병합 (가로로)
        self.data = np.column_stack(results).astype(np.float32)

    def _ibm_to_float(self, data: bytes, count: int) -> np.ndarray:
        """IBM float를 IEEE float로 변환합니다."""
        result = np.zeros(count, dtype=np.float32)

        for i in range(count):
            ibm_bytes = data[i*4:(i+1)*4]
            if len(ibm_bytes) < 4:
                break

            ibm_int = struct.unpack('>I', ibm_bytes)[0]

            # IBM float 파싱
            sign = (ibm_int >> 31) & 1
            exponent = (ibm_int >> 24) & 0x7f
            mantissa = ibm_int & 0x00ffffff

            # IEEE float로 변환
            if mantissa == 0:
                result[i] = 0.0
            else:
                value = mantissa / 16777216.0  # 2^24
                value *= 16.0 ** (exponent - 64)
                if sign:
                    value = -value
                result[i] = value

        return result

    def _normalize_data(self):
        """데이터를 정규화합니다 (각 트레이스별로)"""
        if self.data is not None:
            # 각 트레이스별로 정규화 (열 방향)
            for i in range(self.num_traces):
                trace = self.data[:, i]
                max_abs = np.max(np.abs(trace))
                if max_abs > 0:
                    self.data[:, i] = trace / max_abs

    def get_data(self) -> Optional[np.ndarray]:
        """
        정규화된 데이터를 반환합니다.

        Returns:
            데이터 배열 (num_samples x num_traces) - 가로 방향
        """
        return self.data

    def get_raw_data(self) -> Optional[np.ndarray]:
        """
        원본 데이터를 반환합니다 (정규화 전).

        Returns:
            원본 데이터 배열 (num_samples x num_traces)
        """
        return self.raw_data

    def get_trace(self, trace_index: int) -> Optional[np.ndarray]:
        """
        특정 트레이스 데이터를 반환합니다.

        Args:
            trace_index: 트레이스 인덱스

        Returns:
            트레이스 데이터
        """
        if self.data is not None and 0 <= trace_index < self.num_traces:
            return self.data[:, trace_index]
        return None

    def get_dimensions(self) -> Tuple[int, int]:
        """
        데이터 차원을 반환합니다.

        Returns:
            (num_samples, num_traces) - 가로 방향
        """
        return (self.num_samples, self.num_traces)

    def get_sample_rate(self) -> float:
        """
        샘플링 레이트를 반환합니다.

        Returns:
            샘플링 레이트 (초 단위)
        """
        return self.sample_rate

    def get_time_axis(self) -> np.ndarray:
        """
        시간 축 데이터를 반환합니다.

        Returns:
            시간 배열 (초 단위)
        """
        return np.arange(self.num_samples) * self.sample_rate

    def close(self):
        """파일을 닫습니다."""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None

        if self.file_obj is not None:
            self.file_obj.close()
            self.file_obj = None


def _load_trace_worker(filename: str, trace_idx: int, num_samples: int,
                       header_size: int, trace_size: int) -> np.ndarray:
    """
    병렬 처리를 위한 트레이스 로딩 워커 함수.

    Args:
        filename: SEG-Y 파일 경로
        trace_idx: 트레이스 인덱스
        num_samples: 샘플 수
        header_size: 파일 헤더 크기
        trace_size: 트레이스 크기

    Returns:
        트레이스 데이터
    """
    with open(filename, 'rb') as f:
        trace_offset = header_size + trace_idx * trace_size + 240
        f.seek(trace_offset)
        trace_bytes = f.read(num_samples * 4)

        # IBM float to IEEE float
        trace_data = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            ibm_bytes = trace_bytes[i*4:(i+1)*4]
            if len(ibm_bytes) < 4:
                break

            ibm_int = struct.unpack('>I', ibm_bytes)[0]

            # IBM float 파싱
            sign = (ibm_int >> 31) & 1
            exponent = (ibm_int >> 24) & 0x7f
            mantissa = ibm_int & 0x00ffffff

            # IEEE float로 변환
            if mantissa == 0:
                trace_data[i] = 0.0
            else:
                value = mantissa / 16777216.0  # 2^24
                value *= 16.0 ** (exponent - 64)
                if sign:
                    value = -value
                trace_data[i] = value

        return trace_data
