"""
Automatic Gain Control (AGC) for seismic data
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


class AGC:
    """Automatic Gain Control processor"""

    @staticmethod
    def apply(data: np.ndarray, window_length: int = 100, method: str = 'rms') -> np.ndarray:
        """
        Apply AGC to seismic data

        Args:
            data: Input data (num_samples x num_traces)
            window_length: AGC window length in samples
            method: 'rms' or 'mean'

        Returns:
            AGC-processed data
        """
        if data is None or data.size == 0:
            return data

        num_samples, num_traces = data.shape
        output = np.zeros_like(data, dtype=np.float32)

        # Process each trace
        for trace_idx in range(num_traces):
            trace = data[:, trace_idx]
            output[:, trace_idx] = AGC._apply_to_trace(trace, window_length, method)

        return output

    @staticmethod
    def _apply_to_trace(trace: np.ndarray, window_length: int, method: str) -> np.ndarray:
        """Apply AGC to a single trace"""
        n = len(trace)
        output = np.zeros_like(trace)

        # Calculate half window
        half_window = window_length // 2

        if method == 'rms':
            # RMS-based AGC
            trace_squared = trace ** 2

            # Calculate running RMS using convolution
            window = np.ones(window_length) / window_length
            rms_squared = uniform_filter1d(trace_squared, window_length, mode='nearest')
            rms = np.sqrt(rms_squared)

            # Avoid division by zero
            rms = np.where(rms < 1e-10, 1e-10, rms)

            # Normalize
            output = trace / rms

        elif method == 'mean':
            # Mean absolute value AGC
            abs_trace = np.abs(trace)

            # Calculate running mean
            mean_val = uniform_filter1d(abs_trace, window_length, mode='nearest')

            # Avoid division by zero
            mean_val = np.where(mean_val < 1e-10, 1e-10, mean_val)

            # Normalize
            output = trace / mean_val

        return output

    @staticmethod
    def apply_with_taper(data: np.ndarray, window_length: int = 100,
                        method: str = 'rms', taper_length: int = 10) -> np.ndarray:
        """
        Apply AGC with edge tapering

        Args:
            data: Input data (num_samples x num_traces)
            window_length: AGC window length in samples
            method: 'rms' or 'mean'
            taper_length: Taper length at edges

        Returns:
            AGC-processed data with tapered edges
        """
        # Apply AGC
        agc_data = AGC.apply(data, window_length, method)

        # Apply taper at top and bottom
        if taper_length > 0:
            num_samples = agc_data.shape[0]

            # Top taper
            for i in range(min(taper_length, num_samples)):
                factor = i / taper_length
                agc_data[i, :] = data[i, :] * (1 - factor) + agc_data[i, :] * factor

            # Bottom taper
            for i in range(max(0, num_samples - taper_length), num_samples):
                factor = (num_samples - 1 - i) / taper_length
                idx = i
                agc_data[idx, :] = data[idx, :] * (1 - factor) + agc_data[idx, :] * factor

        return agc_data


class GainControl:
    """Various gain control methods"""

    @staticmethod
    def normalize_trace(data: np.ndarray) -> np.ndarray:
        """
        Normalize each trace independently to [-1, 1]

        Args:
            data: Input data (num_samples x num_traces)

        Returns:
            Normalized data
        """
        num_samples, num_traces = data.shape
        output = np.zeros_like(data)

        for i in range(num_traces):
            trace = data[:, i]
            max_val = np.max(np.abs(trace))
            if max_val > 0:
                output[:, i] = trace / max_val
            else:
                output[:, i] = trace

        return output

    @staticmethod
    def clip_gain(data: np.ndarray, clip_factor: float = 3.0) -> np.ndarray:
        """
        Clip amplitudes beyond a certain threshold

        Args:
            data: Input data
            clip_factor: Clip at clip_factor * std

        Returns:
            Clipped data
        """
        std = np.std(data)
        threshold = clip_factor * std
        return np.clip(data, -threshold, threshold)

    @staticmethod
    def exponential_gain(data: np.ndarray, gain_factor: float = 0.001) -> np.ndarray:
        """
        Apply exponential gain (depth-dependent)

        Args:
            data: Input data (num_samples x num_traces)
            gain_factor: Exponential gain factor

        Returns:
            Gained data
        """
        num_samples = data.shape[0]
        gain = np.exp(np.arange(num_samples) * gain_factor)
        gain = gain.reshape(-1, 1)

        return data * gain
