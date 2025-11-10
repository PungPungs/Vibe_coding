"""
OpenGL 기반 SEG-Y 2D 뷰어 위젯 (Wiggle Trace 방식)
"""
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Optional
from picking_manager import PickingManager


class SegyGLWidget(QOpenGLWidget):
    """OpenGL을 사용한 SEG-Y 2D Wiggle Trace 뷰어 위젯"""

    # 시그널 정의
    mouse_position_changed = pyqtSignal(int, float)  # trace_index, sample_index

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

        # 표시 옵션
        self.show_wiggle: bool = True
        self.show_va: bool = True  # Variable Area
        self.wiggle_scale: float = 5.0
        self.show_picks: bool = True
        self.selected_trace_idx: Optional[int] = None
        self.time_start_ms: float = 0.0
        self.time_end_ms: float = 0.0
        self.sample_rate: float = 0.0

    def set_data(self, data: np.ndarray):
        """
        표시할 데이터를 설정합니다.

        Args:
            data: 정규화된 데이터 배열 (num_samples x num_traces)
        """
        self.data = data
        self.num_samples, self.num_traces = data.shape

        print(f"[GL Widget] Data set: {self.num_samples} samples x {self.num_traces} traces")
        print(f"[GL Widget] Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")

        # 뷰 초기화
        self.reset_view()

        # 재그리기
        self.update()

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
        glClearColor(1.0, 1.0, 1.0, 1.0)  # 흰색 배경
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w: int, h: int):
        """윈도우 크기 변경 시 호출"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Calculate effective sample range for cropping
        start_sample = int(self.time_start_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else 0
        end_sample = int(self.time_end_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else self.num_samples
        
        # Clamp to actual data range
        start_sample = max(0, min(start_sample, self.num_samples))
        end_sample = max(0, min(end_sample, self.num_samples))

        effective_num_samples = end_sample - start_sample
        if effective_num_samples <= 0:
            effective_num_samples = 1 # Prevent division by zero

        aspect = w / h if h != 0 else 1.0
        gluOrtho2D(0, self.num_traces if self.num_traces > 0 else 1,
                   effective_num_samples, 0) # Use effective_num_samples for y-axis
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """OpenGL 렌더링"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glLoadIdentity()

        # if self.data is None:
        #     return

        # # Calculate effective sample range for cropping
        # start_sample = int(self.time_start_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else 0
        # end_sample = int(self.time_end_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else self.num_samples
        
        # # Clamp to actual data range
        # start_sample = max(0, min(start_sample, self.num_samples))
        # end_sample = max(0, min(end_sample, self.num_samples))

        # # 뷰 변환 적용
        # # offset_y를 start_sample만큼 조정하여 렌더링 시작점을 맞춤
        # glTranslatef(self.offset_x, self.offset_y - start_sample, 0.0)
        # glScalef(self.zoom, self.zoom, 1.0)

        # # Wiggle traces 렌더링
        # self._render_wiggle_traces()

        # # 피킹 렌더링
        # if self.show_picks and self.picking_manager is not None:
        #     self._render_picks()

        # # 선택된 트레이스 하이라이트
        # if self.selected_trace_idx is not None:
        #     self._render_selected_trace()

    def _render_wiggle_traces(self):
        """Wiggle trace를 렌더링합니다."""
        if self.data is None:
            return

        # 각 트레이스 렌더링
        for trace_idx in range(self.num_traces):
            trace_data = self.data[:, trace_idx]

            # Variable Area (양수 부분 채우기) - 먼저 그림
            if self.show_va:
                self._render_va(trace_idx, trace_data)

            # Wiggle (선) 그리기
            if self.show_wiggle:
                self._render_wiggle_line(trace_idx, trace_data)

    def _render_selected_trace(self):
        """선택된 트레이스를 하이라이트합니다."""
        if self.selected_trace_idx is None:
            return

        glColor3f(0.0, 0.0, 1.0)  # 파란색
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glVertex2f(self.selected_trace_idx, 0)
        glVertex2f(self.selected_trace_idx, self.num_samples)
        glEnd()

    def _render_wiggle_line(self, trace_idx: int, trace_data: np.ndarray):
        """단일 트레이스의 wiggle 선을 그립니다."""
        glColor3f(0.0, 0.0, 0.0)  # 검은색 선
        glLineWidth(1.0)

        glBegin(GL_LINE_STRIP)
        for sample_idx in range(self.num_samples):
            value = trace_data[sample_idx]
            # 트레이스 중심에서 값만큼 오프셋
            x = trace_idx + value * self.wiggle_scale
            y = sample_idx
            glVertex2f(x, y)
        glEnd()

    def _render_va(self, trace_idx: int, trace_data: np.ndarray):
        """Variable Area (양수 부분)를 채웁니다."""
        glColor4f(0.0, 0.0, 0.0, 0.5)  # 반투명 검은색

        # 양수 영역을 삼각형으로 채우기
        glBegin(GL_TRIANGLES)
        for sample_idx in range(self.num_samples - 1):
            value1 = trace_data[sample_idx]
            value2 = trace_data[sample_idx + 1]

            # 양수 부분만 채우기
            if value1 > 0 or value2 > 0:
                x1 = trace_idx + max(0, value1) * self.wiggle_scale
                x2 = trace_idx + max(0, value2) * self.wiggle_scale
                y1 = sample_idx
                y2 = sample_idx + 1

                # 트레이스 중심선
                glVertex2f(trace_idx, y1)
                glVertex2f(x1, y1)
                glVertex2f(trace_idx, y2)

                glVertex2f(trace_idx, y2)
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
        glEnd()

    def _render_picks(self):
        """피킹을 렌더링합니다."""
        if self.picking_manager is None:
            return

        picks = self.picking_manager.get_picks()
        interpolated = self.picking_manager.get_interpolated_picks()

        if interpolated is None or self.num_traces == 0 or self.num_samples == 0:
            return

        # 보간된 라인 그리기
        glColor3f(1.0, 0.0, 0.0)  # 빨간색
        glLineWidth(3.0)

        glBegin(GL_LINE_STRIP)
        for trace_idx in range(self.num_traces):
            sample_idx = interpolated[trace_idx]
            if sample_idx >= 0:
                glVertex2f(trace_idx, sample_idx)
        glEnd()

        # 피킹 포인트 그리기
        glColor3f(1.0, 0.0, 0.0)  # 빨간색
        glPointSize(8.0)

        glBegin(GL_POINTS)
        for trace_idx, sample_idx in picks:
            glVertex2f(trace_idx, sample_idx)
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

        trace_idx, sample_idx = self._screen_to_data_coords(event.x(), event.y())

        if event.button() == Qt.LeftButton:
            if self.picking_manager is not None and self.picking_manager.is_picking_enabled():
                # 피킹 모드: 피킹 추가
                if trace_idx is not None and sample_idx is not None:
                    self.picking_manager.add_pick(trace_idx, sample_idx)
            else:
                # 피킹 모드가 아니면 트레이스 선택
                if trace_idx is not None:
                    self.selected_trace_idx = trace_idx
                    self.update()

    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.last_mouse_pos is None:
            return

        # 마우스 위치 업데이트
        trace_idx, sample_idx = self._screen_to_data_coords(event.x(), event.y())
        if trace_idx is not None and sample_idx is not None:
            self.mouse_position_changed.emit(trace_idx, sample_idx)

        self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """마우스 버튼 릴리즈 이벤트"""


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

        # Calculate effective sample range for cropping
        start_sample = int(self.time_start_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else 0
        end_sample = int(self.time_end_ms / (self.sample_rate * 1000.0)) if self.sample_rate > 0 else self.num_samples
        
        # Clamp to actual data range
        start_sample = max(0, min(start_sample, self.num_samples))
        end_sample = max(0, min(end_sample, self.num_samples))

        effective_num_samples = end_sample - start_sample
        if effective_num_samples <= 0:
            effective_num_samples = 1 # Prevent division by zero

        # 화면 좌표를 정규화된 좌표로 변환 (크롭된 뷰 기준)
        norm_x = screen_x / self.width()
        norm_y = screen_y / self.height()

        # 데이터 좌표로 변환 (크롭된 뷰 기준)
        trace_idx = int((norm_x * self.num_traces - self.offset_x) / self.zoom)
        sample_idx_cropped = (norm_y * effective_num_samples - self.offset_y) / self.zoom
        sample_idx = sample_idx_cropped + start_sample # Adjust to full data sample index

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
        컬러맵을 설정합니다 (Wiggle trace에서는 사용 안 함).

        Args:
            colormap: 'seismic' 또는 'grayscale'
        """
        # Wiggle trace는 검은색/흰색만 사용
        pass

    def set_show_picks(self, show: bool):
        """
        피킹 표시 여부를 설정합니다.

        Args:
            show: 표시 여부
        """
        self.show_picks = show
        self.update()

    def set_wiggle_scale(self, scale: float):
        """
        Wiggle 스케일을 설정합니다.

        Args:
            scale: 스케일 값 (0.1 ~ 2.0)
        """
        self.wiggle_scale = max(0.1, min(2.0, scale))
        self.update()

    def set_show_wiggle(self, show: bool):
        """Wiggle 선 표시 여부"""
        self.show_wiggle = show
        self.update()

    def set_show_va(self, show: bool):
        """Variable Area 표시 여부"""
        self.show_va = show
        self.update()

    def set_time_cropping(self, time_start_ms: float, time_end_ms: float):
        """
        시간 자르기 범위를 설정합니다.

        Args:
            time_start_ms: 시작 시간 (밀리초)
            time_end_ms: 종료 시간 (밀리초)
        """
        self.time_start_ms = time_start_ms
        self.time_end_ms = time_end_ms
        self.update()
