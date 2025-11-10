"""
Colormap system for seismic data visualization
"""
import numpy as np
from enum import Enum


class ColormapType(Enum):
    """Colormap types"""
    SEISMIC = "Seismic"
    GRAYSCALE = "Grayscale"
    JET = "Jet"
    VIRIDIS = "Viridis"
    RED_WHITE_BLUE = "Red-White-Blue"
    BROWN_WHITE_GREEN = "Brown-White-Green"


class Colormap:
    """Colormap generator for seismic data"""

    @staticmethod
    def get_colormap(colormap_type: ColormapType, n_colors: int = 256) -> np.ndarray:
        """
        Generate colormap array

        Args:
            colormap_type: Type of colormap
            n_colors: Number of color levels

        Returns:
            RGB colormap array (n_colors x 3)
        """
        if colormap_type == ColormapType.SEISMIC:
            return Colormap._seismic(n_colors)
        elif colormap_type == ColormapType.GRAYSCALE:
            return Colormap._grayscale(n_colors)
        elif colormap_type == ColormapType.JET:
            return Colormap._jet(n_colors)
        elif colormap_type == ColormapType.VIRIDIS:
            return Colormap._viridis(n_colors)
        elif colormap_type == ColormapType.RED_WHITE_BLUE:
            return Colormap._red_white_blue(n_colors)
        elif colormap_type == ColormapType.BROWN_WHITE_GREEN:
            return Colormap._brown_white_green(n_colors)
        else:
            return Colormap._seismic(n_colors)

    @staticmethod
    def _seismic(n: int) -> np.ndarray:
        """Seismic colormap: blue-white-red"""
        colors = np.zeros((n, 3))
        half = n // 2

        # Blue to white
        colors[:half, 2] = np.linspace(1.0, 1.0, half)  # B
        colors[:half, 0] = np.linspace(0.0, 1.0, half)  # R
        colors[:half, 1] = np.linspace(0.0, 1.0, half)  # G

        # White to red
        colors[half:, 0] = np.linspace(1.0, 1.0, n - half)  # R
        colors[half:, 1] = np.linspace(1.0, 0.0, n - half)  # G
        colors[half:, 2] = np.linspace(1.0, 0.0, n - half)  # B

        return colors

    @staticmethod
    def _grayscale(n: int) -> np.ndarray:
        """Grayscale colormap: black-white"""
        gray = np.linspace(0, 1, n)
        return np.column_stack([gray, gray, gray])

    @staticmethod
    def _jet(n: int) -> np.ndarray:
        """Jet colormap: blue-cyan-yellow-red"""
        colors = np.zeros((n, 3))

        # Define control points
        x = np.linspace(0, 1, n)

        # Red channel
        colors[:, 0] = np.interp(x, [0, 0.35, 0.66, 0.89, 1.0],
                                  [0, 0, 1, 1, 0.5])

        # Green channel
        colors[:, 1] = np.interp(x, [0, 0.125, 0.375, 0.64, 0.91, 1.0],
                                  [0, 0, 1, 1, 0, 0])

        # Blue channel
        colors[:, 2] = np.interp(x, [0, 0.11, 0.34, 0.65, 1.0],
                                  [0.5, 1, 1, 0, 0])

        return colors

    @staticmethod
    def _viridis(n: int) -> np.ndarray:
        """Viridis colormap approximation"""
        colors = np.zeros((n, 3))
        x = np.linspace(0, 1, n)

        # Approximation of viridis
        colors[:, 0] = np.interp(x, [0, 0.5, 1], [0.267, 0.127, 0.993])  # R
        colors[:, 1] = np.interp(x, [0, 0.5, 1], [0.005, 0.566, 0.906])  # G
        colors[:, 2] = np.interp(x, [0, 0.5, 1], [0.329, 0.551, 0.144])  # B

        return colors

    @staticmethod
    def _red_white_blue(n: int) -> np.ndarray:
        """Red-White-Blue colormap"""
        colors = np.zeros((n, 3))
        half = n // 2

        # Red to white
        colors[:half, 0] = 1.0
        colors[:half, 1] = np.linspace(0, 1, half)
        colors[:half, 2] = np.linspace(0, 1, half)

        # White to blue
        colors[half:, 0] = np.linspace(1, 0, n - half)
        colors[half:, 1] = np.linspace(1, 0, n - half)
        colors[half:, 2] = 1.0

        return colors

    @staticmethod
    def _brown_white_green(n: int) -> np.ndarray:
        """Brown-White-Green colormap (for geology)"""
        colors = np.zeros((n, 3))
        half = n // 2

        # Brown to white
        colors[:half, 0] = np.linspace(0.6, 1.0, half)
        colors[:half, 1] = np.linspace(0.3, 1.0, half)
        colors[:half, 2] = np.linspace(0.0, 1.0, half)

        # White to green
        colors[half:, 0] = np.linspace(1.0, 0.0, n - half)
        colors[half:, 1] = np.linspace(1.0, 0.6, n - half)
        colors[half:, 2] = np.linspace(1.0, 0.0, n - half)

        return colors

    @staticmethod
    def apply_colormap(data: np.ndarray, colormap: np.ndarray,
                      vmin: float = None, vmax: float = None) -> np.ndarray:
        """
        Apply colormap to data

        Args:
            data: 2D data array
            colormap: Colormap array (n_colors x 3)
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling

        Returns:
            RGB image (height x width x 3)
        """
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # Normalize data to [0, 1]
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)

        normalized = np.clip(normalized, 0, 1)

        # Map to colormap indices
        n_colors = len(colormap)
        indices = (normalized * (n_colors - 1)).astype(np.int32)
        indices = np.clip(indices, 0, n_colors - 1)

        # Apply colormap
        rgb_image = colormap[indices]

        return rgb_image
