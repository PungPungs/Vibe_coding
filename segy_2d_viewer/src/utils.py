"""
유틸리티 함수 모듈
"""
import numpy as np
from typing import Tuple


def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    픽셀 좌표를 정규화된 좌표로 변환합니다 (-1.0 to 1.0).

    Args:
        x: X 좌표 (픽셀)
        y: Y 좌표 (픽셀)
        width: 화면 너비
        height: 화면 높이

    Returns:
        정규화된 (x, y) 좌표
    """
    norm_x = (2.0 * x / width) - 1.0
    norm_y = 1.0 - (2.0 * y / height)
    return norm_x, norm_y


def apply_colormap(value: float, colormap: str = 'seismic') -> Tuple[float, float, float]:
    """
    값에 컬러맵을 적용합니다.

    Args:
        value: 정규화된 값 (-1.0 to 1.0)
        colormap: 컬러맵 이름

    Returns:
        (R, G, B) 색상 값 (0.0 to 1.0)
    """
    if colormap == 'seismic':
        # Seismic colormap: blue for negative, white for zero, red for positive
        if value < 0:
            # Blue to white
            t = (value + 1.0) / 1.0
            return (t, t, 1.0)
        else:
            # White to red
            t = 1.0 - value
            return (1.0, t, t)
    elif colormap == 'grayscale':
        # Simple grayscale
        v = (value + 1.0) / 2.0
        return (v, v, v)
    else:
        # Default: grayscale
        v = (value + 1.0) / 2.0
        return (v, v, v)


def interpolate_picks(picks: list, num_traces: int) -> np.ndarray:
    """
    피킹된 점들을 선형 보간합니다.

    Args:
        picks: [(trace_index, sample_index), ...] 형태의 피킹 리스트
        num_traces: 전체 트레이스 수

    Returns:
        각 트레이스에 대한 보간된 샘플 인덱스 배열
    """
    if len(picks) == 0:
        return np.full(num_traces, -1, dtype=np.float32)

    picks_sorted = sorted(picks, key=lambda p: p[0])
    result = np.full(num_traces, -1, dtype=np.float32)

    if len(picks_sorted) == 1:
        trace_idx, sample_idx = picks_sorted[0]
        result[trace_idx] = sample_idx
        return result

    # 선형 보간
    for i in range(len(picks_sorted) - 1):
        trace1, sample1 = picks_sorted[i]
        trace2, sample2 = picks_sorted[i + 1]

        for trace_idx in range(trace1, trace2 + 1):
            if trace2 != trace1:
                t = (trace_idx - trace1) / (trace2 - trace1)
                result[trace_idx] = sample1 + t * (sample2 - sample1)
            else:
                result[trace_idx] = sample1

    return result


def save_picks_to_file(picks: list, filename: str):
    """
    피킹 데이터를 파일로 저장합니다.

    Args:
        picks: [(trace_index, sample_index), ...] 형태의 피킹 리스트
        filename: 저장할 파일 경로
    """
    with open(filename, 'w') as f:
        f.write("Trace,Sample,Time\n")
        for trace_idx, sample_idx in picks:
            f.write(f"{trace_idx},{sample_idx},{sample_idx}\n")


def load_picks_from_file(filename: str) -> list:
    """
    파일에서 피킹 데이터를 로드합니다.

    Args:
        filename: 파일 경로

    Returns:
        [(trace_index, sample_index), ...] 형태의 피킹 리스트
    """
    picks = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    trace_idx = int(parts[0])
                    sample_idx = float(parts[1])
                    picks.append((trace_idx, sample_idx))
    except Exception as e:
        print(f"Error loading picks: {e}")

    return picks
