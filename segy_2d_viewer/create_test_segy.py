"""
테스트용 SEG-Y 파일 생성 스크립트
"""
import numpy as np
import segyio


def create_test_segy(filename='test_data.sgy', num_traces=100, num_samples=500):
    """
    테스트용 SEG-Y 파일을 생성합니다.

    Args:
        filename: 출력 파일명
        num_traces: 트레이스 수
        num_samples: 샘플 수
    """
    spec = segyio.spec()
    spec.format = 1  # 4-byte IBM float
    spec.sorting = 1  # Inline sorting
    spec.samples = np.arange(num_samples) * 2  # 2ms sampling
    spec.ilines = np.arange(1, num_traces + 1)
    spec.xlines = np.array([1])

    print(f"Creating test SEG-Y file: {filename}")
    print(f"  Traces: {num_traces}")
    print(f"  Samples: {num_samples}")
    print(f"  Sample rate: 2ms")

    with segyio.create(filename, spec) as f:
        # 각 트레이스에 대해 합성 지진파 데이터 생성
        for i in range(num_traces):
            # 랜덤 진폭의 sinusoid 생성
            t = np.arange(num_samples) * 0.002  # 시간 (초)

            # First break를 시뮬레이션 (선형 증가)
            first_break_time = 0.05 + i * 0.001  # 50ms + 트레이스당 1ms 증가
            first_break_sample = int(first_break_time / 0.002)

            # 지진파 데이터 생성
            trace = np.zeros(num_samples)

            # First break 이후에만 신호 추가
            if first_break_sample < num_samples:
                # Ricker wavelet 생성
                f0 = 25  # 주파수 (Hz)
                for j in range(first_break_sample, num_samples):
                    t_rel = (j - first_break_sample) * 0.002
                    trace[j] = (1 - 2 * (np.pi * f0 * t_rel)**2) * \
                               np.exp(-(np.pi * f0 * t_rel)**2)

                # 노이즈 추가
                trace += np.random.randn(num_samples) * 0.1

            # 트레이스 저장
            f.trace[i] = trace

            # 헤더 설정
            f.header[i] = {
                segyio.su.tracl: i + 1,
                segyio.su.tracr: i + 1,
                segyio.su.fldr: 1,
                segyio.su.tracf: i + 1,
                segyio.su.cdp: i + 1,
                segyio.su.trid: 1,
            }

    print(f"Test SEG-Y file created successfully: {filename}")


if __name__ == '__main__':
    import os

    # resources 디렉토리 생성
    os.makedirs('resources', exist_ok=True)

    # 기본 테스트 파일 생성
    create_test_segy('resources/test_data.sgy', num_traces=100, num_samples=500)

    # 더 큰 테스트 파일 생성 (옵션)
    # create_test_segy('resources/test_data_large.sgy', num_traces=500, num_samples=1000)
