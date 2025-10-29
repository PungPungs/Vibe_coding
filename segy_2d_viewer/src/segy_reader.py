"""
SEG-Y 파일 읽기 및 파싱 모듈
"""
import numpy as np
import segyio
from typing import Optional, Tuple


class SegyReader:
    """SEG-Y 파일을 읽고 데이터를 처리하는 클래스"""

    def __init__(self):
        self.filename: Optional[str] = None
        self.segy_file: Optional[segyio.SegyFile] = None
        self.data: Optional[np.ndarray] = None
        self.num_traces: int = 0
        self.num_samples: int = 0
        self.sample_rate: float = 0.0

    def load_file(self, filename: str) -> bool:
        """
        SEG-Y 파일을 로드합니다.

        Args:
            filename: SEG-Y 파일 경로

        Returns:
            성공 여부
        """
        try:
            self.filename = filename
            self.segy_file = segyio.open(filename, ignore_geometry=True)

            # 기본 정보 읽기
            self.num_traces = len(self.segy_file.trace)
            self.num_samples = len(self.segy_file.samples)
            self.sample_rate = segyio.tools.dt(self.segy_file) / 1000.0  # ms to seconds

            # 데이터 읽기
            self.data = np.zeros((self.num_traces, self.num_samples), dtype=np.float32)
            for i, trace in enumerate(self.segy_file.trace):
                self.data[i, :] = trace

            # 데이터 정규화 (시각화를 위해)
            self._normalize_data()

            return True

        except Exception as e:
            print(f"Error loading SEG-Y file: {e}")
            return False

    def _normalize_data(self):
        """데이터를 정규화합니다 (시각화를 위해)"""
        if self.data is not None:
            # 각 트레이스별로 정규화
            for i in range(self.num_traces):
                trace = self.data[i, :]
                max_abs = np.max(np.abs(trace))
                if max_abs > 0:
                    self.data[i, :] = trace / max_abs

    def get_data(self) -> Optional[np.ndarray]:
        """
        정규화된 데이터를 반환합니다.

        Returns:
            데이터 배열 (num_traces x num_samples)
        """
        return self.data

    def get_trace(self, trace_index: int) -> Optional[np.ndarray]:
        """
        특정 트레이스 데이터를 반환합니다.

        Args:
            trace_index: 트레이스 인덱스

        Returns:
            트레이스 데이터
        """
        if self.data is not None and 0 <= trace_index < self.num_traces:
            return self.data[trace_index, :]
        return None

    def get_dimensions(self) -> Tuple[int, int]:
        """
        데이터 차원을 반환합니다.

        Returns:
            (num_traces, num_samples)
        """
        return (self.num_traces, self.num_samples)

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
        if self.segy_file is not None:
            return self.segy_file.samples / 1000.0  # ms to seconds
        return np.array([])

    def close(self):
        """파일을 닫습니다."""
        if self.segy_file is not None:
            self.segy_file.close()
            self.segy_file = None
