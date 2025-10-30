"""
OpenGL 기반 SEG-Y 2D 뷰어 위젯
"""
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Optional
from picking_manager import PickingManager


class SegyGLWidget(QOpenGLWidget):
    """OpenGL을 사용한 SEG-Y 2D 뷰어 위젯"""

    # 시그널 정의
    mouse_position_changed = pyqtSignal(int, float)  # trace_index, time

    def __init__(self, parent=None):
        super().__init__(parent)

        # 데이터
        self.data: Optional[np.ndarray] = None
        self.num_traces: int = 0
        self.num_samples: int = 0

        # 피킹 매니저
        self.picking_manager: Optional[PickingManager] = None

        # 뷰 파라미터
        self.zoom: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

        # 마우스 상태
        self.last_mouse_pos: Optional[QPoint] = None
        self.is_panning: bool = False

        # 컬러맵
        self.colormap: str = 'seismic'

        # 피킹 표시
        self.show_picks: bool = True

        # OpenGL 텍스처
        self.texture_id: Optional[int] = None
        self.texture_data: Optional[np.ndarray] = None

    def set_data(self, data: np.ndarray):
        """
        표시할 데이터를 설정합니다.

        Args:
            data: 정규화된 데이터 배열 (num_samples x num_traces)
        """
        self.data = data
        self.num_samples, self.num_traces = data.shape

        # 텍스처 데이터 생성
        self._create_texture_data()

        # 뷰 초기화
        self.reset_view()

        # 재그리기
        self.update()

    def _create_texture_data(self):
        """데이터를 OpenGL 텍스처로 변환합니다."""
        if self.data is None:
            return

        # RGB 이미지 생성 (num_samples x num_traces -> height x width)
        height, width = self.data.shape
        self.texture_data = np.zeros((height, width, 3), dtype=np.uint8)

        # 벡터화된 컬러맵 적용 (성능 향상)
        for i in range(height):
            for j in range(width):
                value = self.data[i, j]
                color = self._apply_colormap(value)
                self.texture_data[i, j] = color

    def _apply_colormap(self, value: float) -> tuple:
        """
        값에 컬러맵을 적용합니다.

        Args:
            value: 정규화된 값 (-1.0 to 1.0)

        Returns:
            (R, G, B) 색상 값 (0 to 255)
        """
        # NaN/Inf 처리
        if not np.isfinite(value):
            return (0, 0, 0)

        # 값 클리핑
        value = np.clip(value, -1.0, 1.0)

        if self.colormap == 'seismic':
            if value < 0:
                # Blue to white
                t = (value + 1.0)
                r = int(t * 255)
                g = int(t * 255)
                b = 255
            else:
                # White to red
                t = 1.0 - value
                r = 255
                g = int(t * 255)
                b = int(t * 255)
            return (r, g, b)
        elif self.colormap == 'grayscale':
            v = int(((value + 1.0) / 2.0) * 255)
            return (v, v, v)
        else:
            v = int(((value + 1.0) / 2.0) * 255)
            return (v, v, v)

    def set_picking_manager(self, picking_manager: PickingManager):
        """
        피킹 매니저를 설정합니다.

        Args:
            picking_manager: PickingManager 인스턴스
        """
        self.picking_manager = picking_manager
        self.picking_manager.picks_changed.connect(self.update)

    def initializeGL(self):
        """OpenGL 초기화"""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)

        # 텍스처 생성
        self.texture_id = glGenTextures(1)

    def resizeGL(self, w: int, h: int):
        """윈도우 크기 변경 시 호출"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h != 0 else 1.0
        gluOrtho2D(-aspect, aspect, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """OpenGL 렌더링"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 뷰 변환 적용
        glTranslatef(self.offset_x, self.offset_y, 0.0)
        glScalef(self.zoom, self.zoom, 1.0)

        # 데이터 렌더링
        if self.texture_data is not None:
            self._render_seismic_data()

        # 피킹 렌더링
        if self.show_picks and self.picking_manager is not None:
            self._render_picks()

    def _render_seismic_data(self):
        """지진파 데이터를 렌더링합니다."""
        if self.texture_data is None:
            return

        # 텍스처 업데이트 (width=num_traces, height=num_samples)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.num_traces, self.num_samples,
                     0, GL_RGB, GL_UNSIGNED_BYTE, self.texture_data)

        # 사각형 렌더링 (가로로 표시)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def _render_picks(self):
        """피킹을 렌더링합니다."""
        if self.picking_manager is None:
            return

        picks = self.picking_manager.get_picks()
        interpolated = self.picking_manager.get_interpolated_picks()

        if interpolated is None or self.num_traces == 0 or self.num_samples == 0:
            return

        # 보간된 라인 그리기
        glDisable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glLineWidth(2.0)

        glBegin(GL_LINE_STRIP)
        for trace_idx in range(self.num_traces):
            sample_idx = interpolated[trace_idx]
            if sample_idx >= 0:
                # 정규화된 좌표로 변환 (가로 방향)
                x = -1.0 + 2.0 * trace_idx / self.num_traces
                y = -1.0 + 2.0 * sample_idx / self.num_samples
                glVertex2f(x, y)
        glEnd()

        # 피킹 포인트 그리기
        glColor3f(1.0, 0.0, 0.0)  # Red
        glPointSize(8.0)

        glBegin(GL_POINTS)
        for trace_idx, sample_idx in picks:
            x = -1.0 + 2.0 * trace_idx / self.num_traces
            y = -1.0 + 2.0 * sample_idx / self.num_samples
            glVertex2f(x, y)
        glEnd()

    def wheelEvent(self, event):
        """마우스 휠 이벤트 (확대/축소)"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))  # 줌 범위 제한

        self.update()

    def mousePressEvent(self, event):
        """마우스 버튼 눌림 이벤트"""
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.LeftButton:
            # 피킹 모드
            if self.picking_manager is not None and self.picking_manager.is_picking_enabled():
                trace_idx, sample_idx = self._screen_to_data_coords(event.x(), event.y())
                if trace_idx is not None and sample_idx is not None:
                    self.picking_manager.add_pick(trace_idx, sample_idx)
        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            # 패닝 모드
            self.is_panning = True

    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.last_mouse_pos is None:
            return

        if self.is_panning:
            # 패닝
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            # 화면 좌표를 정규화된 좌표로 변환
            aspect = self.width() / self.height() if self.height() != 0 else 1.0
            self.offset_x += (dx / self.width()) * 2.0 * aspect / self.zoom
            self.offset_y -= (dy / self.height()) * 2.0 / self.zoom

            self.update()
        else:
            # 마우스 위치 업데이트
            trace_idx, sample_idx = self._screen_to_data_coords(event.x(), event.y())
            if trace_idx is not None and sample_idx is not None:
                self.mouse_position_changed.emit(trace_idx, sample_idx)

        self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """마우스 버튼 릴리즈 이벤트"""
        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self.is_panning = False

    def _screen_to_data_coords(self, screen_x: int, screen_y: int) -> tuple:
        """
        화면 좌표를 데이터 좌표로 변환합니다.

        Args:
            screen_x: 화면 X 좌표
            screen_y: 화면 Y 좌표

        Returns:
            (trace_index, sample_index) 또는 (None, None)
        """
        if self.num_traces == 0 or self.num_samples == 0:
            return (None, None)

        # 화면 좌표를 정규화된 좌표로 변환
        aspect = self.width() / self.height() if self.height() != 0 else 1.0
        norm_x = ((screen_x / self.width()) * 2.0 - 1.0) * aspect
        norm_y = 1.0 - (screen_y / self.height()) * 2.0

        # 뷰 변환 역변환
        norm_x = (norm_x - self.offset_x) / self.zoom
        norm_y = (norm_y - self.offset_y) / self.zoom

        # 데이터 좌표로 변환 (가로 방향)
        trace_idx = int((norm_x + 1.0) / 2.0 * self.num_traces)
        sample_idx = (norm_y + 1.0) / 2.0 * self.num_samples

        # 범위 체크
        if 0 <= trace_idx < self.num_traces and 0 <= sample_idx < self.num_samples:
            return (trace_idx, sample_idx)

        return (None, None)

    def reset_view(self):
        """뷰를 초기 상태로 리셋합니다."""
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.update()

    def set_colormap(self, colormap: str):
        """
        컬러맵을 설정합니다.

        Args:
            colormap: 'seismic' 또는 'grayscale'
        """
        self.colormap = colormap
        self._create_texture_data()
        self.update()

    def set_show_picks(self, show: bool):
        """
        피킹 표시 여부를 설정합니다.

        Args:
            show: 표시 여부
        """
        self.show_picks = show
        self.update()
