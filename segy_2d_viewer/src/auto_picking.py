"""
자동 first break picking 알고리즘 모듈
"""
import numpy as np
from typing import Optional, Tuple, List
from multiprocessing import Pool, cpu_count


class AutoPicker:
    """자동 피킹 알고리즘 클래스"""

    def __init__(self):
        self.algorithm = 'sta_lta'  # 기본 알고리즘

    def pick_all_traces(self, data: np.ndarray, algorithm: str = 'sta_lta',
                       **kwargs) -> List[Tuple[int, float]]:
        """
        모든 트레이스에 대해 자동 피킹을 수행합니다.

        Args:
            data: 지진파 데이터 (num_samples x num_traces)
            algorithm: 사용할 알고리즘 ('sta_lta', 'energy_ratio', 'aic')
            **kwargs: 알고리즘별 파라미터

        Returns:
            [(trace_index, sample_index), ...] 형태의 피킹 리스트
        """
        num_samples, num_traces = data.shape
        picks = []

        # 알고리즘 선택
        if algorithm == 'sta_lta':
            pick_func = self._pick_sta_lta
        elif algorithm == 'energy_ratio':
            pick_func = self._pick_energy_ratio
        elif algorithm == 'aic':
            pick_func = self._pick_aic
        else:
            print(f"Unknown algorithm: {algorithm}, using STA/LTA")
            pick_func = self._pick_sta_lta

        # 병렬 처리
        use_parallel = kwargs.get('use_parallel', True)

        if use_parallel and num_traces > 50:
            # 병렬 처리
            num_processes = min(cpu_count(), 8)
            tasks = [(data[:, i], i, algorithm, kwargs) for i in range(num_traces)]

            with Pool(processes=num_processes) as pool:
                results = pool.starmap(_pick_trace_worker, tasks)

            # 결과 수집
            for trace_idx, sample_idx in results:
                if sample_idx is not None:
                    picks.append((trace_idx, sample_idx))
        else:
            # 순차 처리
            for i in range(num_traces):
                trace = data[:, i]
                sample_idx = pick_func(trace, **kwargs)
                if sample_idx is not None:
                    picks.append((i, sample_idx))

        return picks

    def _pick_sta_lta(self, trace: np.ndarray, **kwargs) -> Optional[float]:
        """
        STA/LTA (Short-Term Average / Long-Term Average) 알고리즘.

        가장 널리 사용되는 first break picking 알고리즘.
        단기 평균과 장기 평균의 비율을 계산하여 신호의 시작점을 찾습니다.

        Args:
            trace: 트레이스 데이터
            **kwargs: sta_window, lta_window, threshold

        Returns:
            피킹된 샘플 인덱스
        """
        sta_window = kwargs.get('sta_window', 5)  # 단기 윈도우 (samples)
        lta_window = kwargs.get('lta_window', 50)  # 장기 윈도우 (samples)
        threshold = kwargs.get('threshold', 3.0)  # 임계값

        if len(trace) < lta_window + sta_window:
            return None

        # 절대값 사용
        trace_abs = np.abs(trace)

        # STA/LTA 계산
        sta_lta = np.zeros(len(trace))

        for i in range(lta_window, len(trace) - sta_window):
            lta = np.mean(trace_abs[i - lta_window:i])
            sta = np.mean(trace_abs[i:i + sta_window])

            if lta > 0:
                sta_lta[i] = sta / lta

        # 임계값을 넘는 첫 번째 지점 찾기
        trigger_indices = np.where(sta_lta > threshold)[0]

        if len(trigger_indices) > 0:
            return float(trigger_indices[0])

        return None

    def _pick_energy_ratio(self, trace: np.ndarray, **kwargs) -> Optional[float]:
        """
        Energy Ratio 알고리즘.

        신호의 에너지 비율을 사용하여 first break를 찾습니다.

        Args:
            trace: 트레이스 데이터
            **kwargs: window, threshold

        Returns:
            피킹된 샘플 인덱스
        """
        window = kwargs.get('window', 20)  # 윈도우 크기
        threshold = kwargs.get('threshold', 0.1)  # 임계값 (전체 에너지의 비율)

        if len(trace) < window:
            return None

        # 에너지 계산 (제곱의 합)
        energy = trace ** 2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]

        if total_energy == 0:
            return None

        # 누적 에너지 비율
        energy_ratio = cumulative_energy / total_energy

        # 임계값을 넘는 첫 번째 지점
        trigger_indices = np.where(energy_ratio > threshold)[0]

        if len(trigger_indices) > 0:
            return float(trigger_indices[0])

        return None

    def _pick_aic(self, trace: np.ndarray, **kwargs) -> Optional[float]:
        """
        AIC (Akaike Information Criterion) 알고리즘.

        통계적 방법으로 신호의 특성 변화 지점을 찾습니다.
        AIC가 최소가 되는 지점이 first break입니다.

        Args:
            trace: 트레이스 데이터
            **kwargs: window

        Returns:
            피킹된 샘플 인덱스
        """
        window = kwargs.get('window', 100)  # 검색 윈도우

        if len(trace) < window:
            return None

        # AIC 계산
        aic = np.zeros(len(trace))

        for k in range(1, min(len(trace) - 1, window)):
            if k < 2 or k > len(trace) - 2:
                continue

            # 앞부분과 뒷부분의 분산 계산
            var1 = np.var(trace[:k])
            var2 = np.var(trace[k:min(k + window, len(trace))])

            if var1 > 0 and var2 > 0:
                aic[k] = k * np.log(var1) + (len(trace) - k) * np.log(var2)
            else:
                aic[k] = np.inf

        # AIC가 최소인 지점 찾기 (앞부분 노이즈 제외)
        start_idx = max(1, int(len(trace) * 0.05))  # 첫 5% 제외
        end_idx = min(len(trace), window)

        valid_aic = aic[start_idx:end_idx]
        if len(valid_aic) > 0 and np.min(valid_aic) < np.inf:
            min_idx = np.argmin(valid_aic) + start_idx
            return float(min_idx)

        return None

    def refine_picks(self, data: np.ndarray, picks: List[Tuple[int, float]],
                    window: int = 10) -> List[Tuple[int, float]]:
        """
        피킹 결과를 정제합니다.

        로컬 최대/최소값을 찾아 피킹 위치를 조정합니다.

        Args:
            data: 지진파 데이터 (num_samples x num_traces)
            picks: 피킹 리스트
            window: 검색 윈도우 크기

        Returns:
            정제된 피킹 리스트
        """
        refined_picks = []

        for trace_idx, sample_idx in picks:
            if trace_idx >= data.shape[1]:
                continue

            trace = data[:, trace_idx]
            sample_idx_int = int(sample_idx)

            # 검색 범위
            start = max(0, sample_idx_int - window)
            end = min(len(trace), sample_idx_int + window)

            if start >= end:
                refined_picks.append((trace_idx, sample_idx))
                continue

            # 윈도우 내에서 최대 절대값 찾기
            window_data = np.abs(trace[start:end])
            local_max_idx = np.argmax(window_data)
            refined_sample_idx = start + local_max_idx

            refined_picks.append((trace_idx, float(refined_sample_idx)))

        return refined_picks


def _pick_trace_worker(trace: np.ndarray, trace_idx: int,
                       algorithm: str, kwargs: dict) -> Tuple[int, Optional[float]]:
    """
    병렬 처리를 위한 트레이스 피킹 워커.

    Args:
        trace: 트레이스 데이터
        trace_idx: 트레이스 인덱스
        algorithm: 알고리즘 이름
        kwargs: 알고리즘 파라미터

    Returns:
        (trace_index, sample_index)
    """
    picker = AutoPicker()

    if algorithm == 'sta_lta':
        sample_idx = picker._pick_sta_lta(trace, **kwargs)
    elif algorithm == 'energy_ratio':
        sample_idx = picker._pick_energy_ratio(trace, **kwargs)
    elif algorithm == 'aic':
        sample_idx = picker._pick_aic(trace, **kwargs)
    else:
        sample_idx = picker._pick_sta_lta(trace, **kwargs)

    return (trace_idx, sample_idx)


def get_algorithm_params(algorithm: str) -> dict:
    """
    알고리즘별 기본 파라미터를 반환합니다.

    Args:
        algorithm: 알고리즘 이름

    Returns:
        기본 파라미터 딕셔너리
    """
    if algorithm == 'sta_lta':
        return {
            'sta_window': 5,
            'lta_window': 50,
            'threshold': 3.0
        }
    elif algorithm == 'energy_ratio':
        return {
            'window': 20,
            'threshold': 0.1
        }
    elif algorithm == 'aic':
        return {
            'window': 100
        }
    else:
        return {}
