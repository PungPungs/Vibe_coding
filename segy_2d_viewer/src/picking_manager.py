"""
First break picking 관리 모듈
"""
import numpy as np
from typing import List, Tuple, Optional
from PyQt5.QtCore import QObject, pyqtSignal


class PickingManager(QObject):
    """피킹 데이터를 관리하는 클래스"""

    # 시그널 정의
    picks_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.picks: List[Tuple[int, float]] = []  # [(trace_index, sample_index), ...]
        self.interpolated_picks: Optional[np.ndarray] = None
        self.num_traces: int = 0
        self.picking_enabled: bool = True

    def set_num_traces(self, num_traces: int):
        """
        전체 트레이스 수를 설정합니다.

        Args:
            num_traces: 트레이스 수
        """
        self.num_traces = num_traces
        self.interpolated_picks = np.full(num_traces, -1, dtype=np.float32)

    def add_pick(self, trace_index: int, sample_index: float):
        """
        피킹 포인트를 추가합니다.

        Args:
            trace_index: 트레이스 인덱스
            sample_index: 샘플 인덱스
        """
        if not self.picking_enabled:
            return

        # 같은 트레이스에 이미 피킹이 있는지 확인
        existing_idx = None
        for i, (trace, _) in enumerate(self.picks):
            if trace == trace_index:
                existing_idx = i
                break

        if existing_idx is not None:
            # 기존 피킹 업데이트
            self.picks[existing_idx] = (trace_index, sample_index)
        else:
            # 새 피킹 추가
            self.picks.append((trace_index, sample_index))

        # 정렬
        self.picks.sort(key=lambda p: p[0])

        # 보간 수행
        self._interpolate()

        # 시그널 발생
        self.picks_changed.emit()

    def remove_pick(self, trace_index: int):
        """
        특정 트레이스의 피킹을 제거합니다.

        Args:
            trace_index: 트레이스 인덱스
        """
        self.picks = [(t, s) for t, s in self.picks if t != trace_index]
        self._interpolate()
        self.picks_changed.emit()

    def clear_picks(self):
        """모든 피킹을 제거합니다."""
        self.picks.clear()
        if self.interpolated_picks is not None:
            self.interpolated_picks.fill(-1)
        self.picks_changed.emit()

    def _interpolate(self):
        """피킹된 포인트들을 선형 보간합니다."""
        if self.num_traces == 0 or self.interpolated_picks is None:
            return

        self.interpolated_picks.fill(-1)

        if len(self.picks) == 0:
            return

        if len(self.picks) == 1:
            trace_idx, sample_idx = self.picks[0]
            self.interpolated_picks[trace_idx] = sample_idx
            return

        # 선형 보간
        for i in range(len(self.picks) - 1):
            trace1, sample1 = self.picks[i]
            trace2, sample2 = self.picks[i + 1]

            for trace_idx in range(trace1, trace2 + 1):
                if trace2 != trace1:
                    t = (trace_idx - trace1) / (trace2 - trace1)
                    self.interpolated_picks[trace_idx] = sample1 + t * (sample2 - sample1)
                else:
                    self.interpolated_picks[trace_idx] = sample1

    def get_picks(self) -> List[Tuple[int, float]]:
        """
        모든 피킹 포인트를 반환합니다.

        Returns:
            [(trace_index, sample_index), ...] 형태의 리스트
        """
        return self.picks.copy()

    def get_interpolated_picks(self) -> Optional[np.ndarray]:
        """
        보간된 피킹 데이터를 반환합니다.

        Returns:
            각 트레이스에 대한 샘플 인덱스 배열
        """
        return self.interpolated_picks

    def get_pick_at_trace(self, trace_index: int) -> Optional[float]:
        """
        특정 트레이스의 피킹 값을 반환합니다.

        Args:
            trace_index: 트레이스 인덱스

        Returns:
            샘플 인덱스 또는 None
        """
        if self.interpolated_picks is not None and 0 <= trace_index < len(self.interpolated_picks):
            value = self.interpolated_picks[trace_index]
            if value >= 0:
                return value
        return None

    def is_picking_enabled(self) -> bool:
        """
        피킹 활성화 상태를 반환합니다.

        Returns:
            피킹 활성화 여부
        """
        return self.picking_enabled

    def set_picking_enabled(self, enabled: bool):
        """
        피킹 활성화 상태를 설정합니다.

        Args:
            enabled: 활성화 여부
        """
        self.picking_enabled = enabled

    def save_to_file(self, filename: str) -> bool:
        """
        피킹 데이터를 파일로 저장합니다.

        Args:
            filename: 파일 경로

        Returns:
            성공 여부
        """
        try:
            with open(filename, 'w') as f:
                f.write("Trace,Sample\n")
                for trace_idx, sample_idx in self.picks:
                    f.write(f"{trace_idx},{sample_idx}\n")
            return True
        except Exception as e:
            print(f"Error saving picks: {e}")
            return False

    def load_from_file(self, filename: str) -> bool:
        """
        파일에서 피킹 데이터를 로드합니다.

        Args:
            filename: 파일 경로

        Returns:
            성공 여부
        """
        try:
            self.picks.clear()
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        trace_idx = int(parts[0])
                        sample_idx = float(parts[1])
                        self.picks.append((trace_idx, sample_idx))

            self.picks.sort(key=lambda p: p[0])
            self._interpolate()
            self.picks_changed.emit()
            return True
        except Exception as e:
            print(f"Error loading picks: {e}")
            return False
