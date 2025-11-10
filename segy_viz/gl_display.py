"""
Advanced OpenGL display widget supporting multiple seismic rendering modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import (
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glFlush,
    glGenTextures,
    glHint,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glScalef,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex2f,
    glViewport,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_FLOAT,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINE_STRIP,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
)
from OpenGL.GLU import gluOrtho2D

from .agc import AGC, GainControl
from .colormap import Colormap, ColormapType
from .display_modes import DisplayMode, DisplaySettings


@dataclass(frozen=True)
class _ViewParams:
    """Convenience container for current viewport settings."""

    trace_start: int
    trace_end: int
    visible_traces: int


class SeismicDisplay(QOpenGLWidget):
    """
    RadEx-style seismic data display widget that mixes wiggle, density, and filled
    render modes. The widget keeps a fixed-width viewport and advances through the
    SEG-Y volume in discrete pages so the screen is always filled.
    """

    mouse_position_changed = pyqtSignal(int, float)  # trace_idx, sample_idx
    trace_selected = pyqtSignal(int)  # global trace index

    MIN_TRACES_PER_WINDOW = 10
    DEFAULT_PIXELS_PER_TRACE = 24
    MAX_ZOOM = 12.0
    MIN_ZOOM = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.raw_data: np.ndarray | None = None
        self.processed_data: np.ndarray | None = None
        self.num_samples = 0
        self.num_traces = 0

        self.settings = DisplaySettings()
        self.settings.trace_start = 0
        self.settings.trace_window = 0

        self.pixels_per_trace = self.DEFAULT_PIXELS_PER_TRACE
        self.traces_per_screen = 0

        self.colormap = Colormap.get_colormap(ColormapType.SEISMIC)
        self.texture_data: np.ndarray | None = None
        self.texture_id: int | None = None
        self._texture_dirty = True

        self.current_scale_factor = 1.0

        self.last_mouse_pos = None
        self.is_panning = False
        self.selected_trace: int | None = None

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def set_data(self, data: np.ndarray) -> None:
        """Assign seismic amplitude data (samples x traces)."""
        if data.ndim != 2:
            raise ValueError("Expected 2D array with shape (samples, traces)")

        # Ensure a float32 view for predictable OpenGL uploads
        self.raw_data = np.asarray(data, dtype=np.float32)
        self.num_samples, self.num_traces = self.raw_data.shape

        self.selected_trace = None
        self.settings.trace_start = 0
        self.settings.zoom = 1.0
        self.settings.offset_x = 0.0
        self.settings.offset_y = 0.0

        self._update_trace_window(force=True)
        self._process_data()

        print(f"[Display] Data set: {self.num_samples} x {self.num_traces}")
        print(f"[Display] Range: [{np.min(self.raw_data):.2e}, {np.max(self.raw_data):.2e}]")
        print(f"[Display] Viewport mode: {self.traces_per_screen} traces per screen")
        print("[Display] Use Left/Right arrows to page traces")

        self._texture_dirty = True
        self.update()

    def _process_data(self) -> None:
        """Apply gain, clipping, and prepare processed buffer."""
        if self.raw_data is None:
            return

        processed = self.raw_data.copy()
        print(f"[Process] Raw data range: [{np.min(processed):.2e}, {np.max(processed):.2e}]")

        if self.settings.use_agc:
            method = self._resolve_gain_method(self.settings.agc_method)
            processed = AGC.apply(processed, self.settings.agc_window, method)
            print(f"[Process] After AGC: [{np.min(processed):.2e}, {np.max(processed):.2e}]")

        if self.settings.clip_percentile < 100.0:
            percentile = self.settings.clip_percentile
            threshold = np.percentile(np.abs(processed), percentile)
            processed = np.clip(processed, -threshold, threshold)
            print(f"[Process] After clipping ({percentile}%): threshold = {threshold:.2e}")

        self.processed_data = processed
        print(f"[Process] Final data range: [{np.min(processed):.2e}, {np.max(processed):.2e}]")
        self._texture_dirty = True

        if self.settings.mode in (DisplayMode.VARIABLE_DENSITY, DisplayMode.WIGGLE_VD):
            self._prepare_texture(self._current_slice())

    def _resolve_gain_method(self, method: str | GainControl) -> GainControl:
        if isinstance(method, GainControl):
            return method
        try:
            return GainControl(method.lower())
        except ValueError:
            return GainControl.RMS

    # ------------------------------------------------------------------
    # View calculations
    # ------------------------------------------------------------------
    def _update_trace_window(self, force: bool = False) -> None:
        """Recompute how many traces are shown per page."""
        if self.num_traces == 0:
            self.settings.trace_window = 0
            self.traces_per_screen = 0
            return

        width = max(self.width(), 800)
        by_pixels = max(self.MIN_TRACES_PER_WINDOW, int(width / self.pixels_per_trace))

        if self.num_traces >= self.MIN_TRACES_PER_WINDOW:
            by_ratio = max(self.MIN_TRACES_PER_WINDOW, self.num_traces // 10)
        else:
            by_ratio = self.num_traces

        new_window = max(1, min(self.num_traces, min(by_pixels, by_ratio)))

        if force or new_window != self.settings.trace_window:
            self.settings.trace_window = new_window
            self.traces_per_screen = new_window
            self._clamp_trace_start()
            self._texture_dirty = True
            print(f"[Display] Viewport adjusted: {new_window} traces per screen")

    def _clamp_trace_start(self) -> None:
        """Ensure the trace start stays inside the available data."""
        max_start = max(self.num_traces - self.settings.trace_window, 0)
        new_start = int(np.clip(self.settings.trace_start, 0, max_start))
        if new_start != self.settings.trace_start:
            self.settings.trace_start = new_start

    def _visible_range(self) -> _ViewParams:
        start = self.settings.trace_start
        end = min(start + self.settings.trace_window, self.num_traces)
        visible = max(0, end - start)
        return _ViewParams(start, end, visible)

    def _current_slice(self) -> np.ndarray:
        """Return the currently visible traces."""
        if self.processed_data is None or self.settings.trace_window == 0:
            return np.empty((0, 0), dtype=np.float32)
        params = self._visible_range()
        if params.visible_traces == 0:
            return np.empty((self.num_samples, 0), dtype=np.float32)
        return self.processed_data[:, params.trace_start:params.trace_end]

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        self.texture_id = glGenTextures(1)

    def resizeGL(self, w: int, h: int) -> None:  # noqa: ARG002
        glViewport(0, 0, w, h)
        if self.num_traces > 0:
            self._update_trace_window()

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        if self.processed_data is None or self.settings.trace_window == 0:
            return

        params = self._visible_range()
        if params.visible_traces == 0:
            return

        # Configure projection to match the number of visible traces
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, params.visible_traces, self.num_samples, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(self.settings.offset_x, self.settings.offset_y, 0.0)
        glScalef(self.settings.zoom, self.settings.zoom, 1.0)

        visible_data = self._current_slice()
        if visible_data.size == 0:
            return

        mode = self.settings.mode
        if mode == DisplayMode.VARIABLE_DENSITY:
            self._render_variable_density(visible_data, params.visible_traces)
        elif mode == DisplayMode.WIGGLE:
            self._render_wiggle(visible_data)
        elif mode == DisplayMode.VARIABLE_AREA:
            self._render_variable_area(visible_data)
        elif mode == DisplayMode.WIGGLE_VA:
            self._render_variable_area(visible_data)
            self._render_wiggle(visible_data)
        elif mode == DisplayMode.WIGGLE_VD:
            self._render_variable_density(visible_data, params.visible_traces)
            self._render_wiggle(visible_data)

        if self.selected_trace is not None:
            self._render_selected_trace(params)

        glFlush()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _prepare_texture(self, visible_data: np.ndarray) -> None:
        if visible_data.size == 0:
            self.texture_data = None
            self._texture_dirty = False
            return

        if not self._texture_dirty and self.texture_data is not None:
            return

        rgb_image = Colormap.apply_colormap(visible_data, self.colormap)
        self.texture_data = np.ascontiguousarray(rgb_image, dtype=np.float32)
        self._texture_dirty = False

    def _render_variable_density(self, visible_data: np.ndarray, visible_traces: int) -> None:
        self._prepare_texture(visible_data)
        if self.texture_data is None or self.texture_id is None:
            return

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        h, w, _ = self.texture_data.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, self.texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glColor4f(1.0, 1.0, 1.0, self.settings.vd_alpha)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(visible_traces, 0)
        glTexCoord2f(1, 1)
        glVertex2f(visible_traces, self.num_samples)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.num_samples)
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def _compute_trace_scale(self, visible_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(visible_data)))
        if rms < 1e-10:
            max_val = np.max(np.abs(visible_data))
            if max_val > 1e-10:
                return max_val
            return 1.0
        return rms * 3.0

    def _render_wiggle(self, visible_data: np.ndarray) -> None:
        scale_basis = self._compute_trace_scale(visible_data)
        self.current_scale_factor = scale_basis

        glColor3f(*self.settings.wiggle_color)
        glLineWidth(self.settings.wiggle_line_width)
        scale = self.settings.wiggle_amplitude / self.current_scale_factor

        samples, trace_count = visible_data.shape
        for display_idx in range(trace_count):
            trace_data = visible_data[:, display_idx]
            glBegin(GL_LINE_STRIP)
            for sample_idx in range(samples):
                value = trace_data[sample_idx] * scale
                x = display_idx + value
                y = sample_idx
                glVertex2f(x, y)
            glEnd()

    def _render_variable_area(self, visible_data: np.ndarray) -> None:
        scale_basis = self._compute_trace_scale(visible_data)
        self.current_scale_factor = scale_basis
        scale = self.settings.wiggle_amplitude / self.current_scale_factor

        glColor4f(*self.settings.va_color)
        samples, trace_count = visible_data.shape
        for display_idx in range(trace_count):
            trace_data = visible_data[:, display_idx]
            glBegin(GL_TRIANGLES)
            for sample_idx in range(samples - 1):
                v1 = trace_data[sample_idx] * scale
                v2 = trace_data[sample_idx + 1] * scale
                if v1 <= 0 and v2 <= 0:
                    continue
                x1 = display_idx + max(0.0, v1)
                x2 = display_idx + max(0.0, v2)
                y1 = sample_idx
                y2 = sample_idx + 1

                glVertex2f(display_idx, y1)
                glVertex2f(x1, y1)
                glVertex2f(display_idx, y2)

                glVertex2f(display_idx, y2)
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
            glEnd()

    def _render_selected_trace(self, params: _ViewParams) -> None:
        assert self.selected_trace is not None
        display_idx = self.selected_trace - params.trace_start
        if display_idx < 0 or display_idx >= params.visible_traces:
            return

        glColor4f(0.0, 0.0, 1.0, 0.3)
        glLineWidth(3.0)

        glBegin(GL_LINES)
        glVertex2f(display_idx + 0.5, 0)
        glVertex2f(display_idx + 0.5, self.num_samples)
        glEnd()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.settings.zoom *= zoom_factor
        self.settings.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.settings.zoom))
        self._clamp_offsets()
        self.update()

    def mousePressEvent(self, event) -> None:
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.LeftButton:
            trace_idx = self._screen_to_trace(event.x())
            if trace_idx is not None:
                self.selected_trace = trace_idx
                self.trace_selected.emit(trace_idx)
                self.update()
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_panning = True

    def mouseMoveEvent(self, event) -> None:
        trace_idx, sample_idx = self._screen_to_data(event.x(), event.y())
        if trace_idx is not None and sample_idx is not None:
            self.mouse_position_changed.emit(trace_idx, sample_idx)

        if self.is_panning and self.last_mouse_pos is not None:
            dy = event.y() - self.last_mouse_pos.y()
            self._adjust_vertical_offset(dy)
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_panning = False
            self.last_mouse_pos = None

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Left:
            self._advance_view(-1)
        elif key == Qt.Key_Right:
            self._advance_view(1)
        elif key == Qt.Key_Up:
            self._adjust_vertical_offset(10.0)
            self.update()
        elif key == Qt.Key_Down:
            self._adjust_vertical_offset(-10.0)
            self.update()
        elif key == Qt.Key_Home:
            self.settings.trace_start = 0
            self._texture_dirty = True
            self.update()
            params = self._visible_range()
            print(f"[Nav] Page: {params.trace_start + 1}-{params.trace_end} / {self.num_traces}")
        elif key == Qt.Key_End:
            self.settings.trace_start = max(self.num_traces - self.settings.trace_window, 0)
            self._texture_dirty = True
            self.update()
            params = self._visible_range()
            print(f"[Nav] Page: {params.trace_start + 1}-{params.trace_end} / {self.num_traces}")
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.settings.zoom = min(self.MAX_ZOOM, self.settings.zoom * 1.2)
            self._clamp_offsets()
            self.update()
        elif key == Qt.Key_Minus:
            self.settings.zoom = max(self.MIN_ZOOM, self.settings.zoom / 1.2)
            self._clamp_offsets()
            self.update()
        elif key == Qt.Key_R:
            self.reset_view()

    def _advance_view(self, step: int) -> None:
        if self.settings.trace_window == 0:
            return
        increment = step * self.settings.trace_window
        new_start = self.settings.trace_start + increment
        max_start = max(self.num_traces - self.settings.trace_window, 0)
        new_start = int(np.clip(new_start, 0, max_start))
        if new_start != self.settings.trace_start:
            self.settings.trace_start = new_start
            self._texture_dirty = True
            params = self._visible_range()
            print(f"[Nav] Page: {params.trace_start + 1}-{params.trace_end} / {self.num_traces}")
            self.update()

    def _adjust_vertical_offset(self, delta_pixels: float) -> None:
        # Negative delta moves view up (earlier samples)
        self.settings.offset_y += -delta_pixels * 0.1
        self._clamp_offsets()

    def _clamp_offsets(self) -> None:
        if self.settings.zoom <= 1.0:
            self.settings.offset_x = 0.0
            self.settings.offset_y = 0.0
            return

        params = self._visible_range()
        horizontal_limit = (1.0 - self.settings.zoom) * params.visible_traces
        vertical_limit = (1.0 - self.settings.zoom) * self.num_samples

        self.settings.offset_x = float(np.clip(self.settings.offset_x, horizontal_limit, 0.0))
        self.settings.offset_y = float(np.clip(self.settings.offset_y, vertical_limit, 0.0))

    def _screen_to_data(self, screen_x: int, screen_y: int) -> tuple[int | None, float | None]:
        if self.processed_data is None or self.settings.trace_window == 0:
            return (None, None)

        params = self._visible_range()
        if params.visible_traces == 0:
            return (None, None)

        norm_x = screen_x / max(self.width(), 1)
        norm_y = screen_y / max(self.height(), 1)

        x = norm_x * params.visible_traces
        y = norm_y * self.num_samples

        x = (x - self.settings.offset_x) / self.settings.zoom
        y = (y - self.settings.offset_y) / self.settings.zoom

        trace_idx = int(np.floor(x)) + params.trace_start
        sample_idx = y

        if 0 <= trace_idx < self.num_traces and 0 <= sample_idx < self.num_samples:
            return (trace_idx, sample_idx)
        return (None, None)

    def _screen_to_trace(self, screen_x: int) -> int | None:
        trace_idx, _ = self._screen_to_data(screen_x, self.height() // 2)
        return trace_idx

    # ------------------------------------------------------------------
    # Settings exposed to UI
    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        self.settings.zoom = 1.0
        self.settings.offset_x = 0.0
        self.settings.offset_y = 0.0
        self.settings.trace_start = 0
        self._texture_dirty = True
        params = self._visible_range()
        print(f"[Reset] View reset, showing traces 1-{params.trace_end}")
        self.update()

    def set_display_mode(self, mode: DisplayMode) -> None:
        self.settings.mode = mode
        if self.processed_data is not None:
            if mode in (DisplayMode.VARIABLE_DENSITY, DisplayMode.WIGGLE_VD):
                self._texture_dirty = True
            self.update()

    def set_colormap(self, colormap_type: ColormapType) -> None:
        self.colormap = Colormap.get_colormap(colormap_type)
        self._texture_dirty = True
        self.update()

    def set_agc(self, enabled: bool, window: int = 100, method: str = "rms") -> None:
        self.settings.use_agc = enabled
        self.settings.agc_window = window
        self.settings.agc_method = method
        self._process_data()
        self.update()

    def set_clip_percentile(self, percentile: float) -> None:
        self.settings.clip_percentile = percentile
        self._process_data()
        self.update()

    def set_wiggle_amplitude(self, amplitude: float) -> None:
        self.settings.wiggle_amplitude = amplitude
        self.update()

    def export_image(self, path) -> None:
        image = self.grabFramebuffer()
        image.save(str(path))

    def has_data(self) -> bool:
        return self.processed_data is not None
