"""
OpenGL Wiggle Trace Viewer
"""
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from OpenGL.GL import *
from OpenGL.GLU import *


class TraceViewer(QOpenGLWidget):
    """OpenGL 기반 Wiggle Trace Viewer"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 데이터
        self.data = None
        self.num_samples = 0
        self.num_traces = 0

        # 뷰 파라미터
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # 표시 범위 (trace 인덱스)
        self.trace_start = 0
        self.trace_window = 100  # 한 번에 보여줄 trace 개수

        # 마우스 상태
        self.last_mouse_pos = None
        self.is_panning = False

        # 데이터 스케일 (자동 계산됨)
        self.data_scale = 1.0
        self.wiggle_amplitude = 0.8  # trace 간격 대비 wiggle 진폭

        # 키보드 포커스 활성화
        self.setFocusPolicy(Qt.StrongFocus)

    def set_data(self, data: np.ndarray):
        """
        데이터 설정 (num_samples x num_traces)

        Args:
            data: SEG-Y 데이터 배열
        """
        self.data = data
        self.num_samples, self.num_traces = data.shape

        # 데이터 스케일 자동 계산 (전체 데이터의 최대 절대값)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            self.data_scale = 1.0 / max_val

        # 초기 표시 범위
        self.trace_start = 0
        self.trace_window = min(100, self.num_traces)

        print(f"[Viewer] Data loaded: {self.num_samples} x {self.num_traces}")
        print(f"[Viewer] Data range: [{np.min(data):.2e}, {np.max(data):.2e}]")
        print(f"[Viewer] Auto scale: {self.data_scale:.2e}")

        self.reset_view()
        self.update()

    def initializeGL(self):
        """OpenGL 초기화"""
        glClearColor(1.0, 1.0, 1.0, 1.0)  # 흰색 배경
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w, h):
        """윈도우 크기 변경"""
        glViewport(0, 0, w, h)

    def paintGL(self):
        """OpenGL 렌더링"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        if self.data is None or self.num_traces == 0:
            return

        # 프로젝션 설정
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # 현재 윈도우에 보여줄 trace 범위
        trace_end = min(self.trace_start + self.trace_window, self.num_traces)
        visible_traces = trace_end - self.trace_start

        if visible_traces <= 0:
            return

        # 좌표계: x=[0, visible_traces], y=[0, num_samples]
        gluOrtho2D(0, visible_traces, self.num_samples, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 뷰 변환 적용
        glTranslatef(self.offset_x, self.offset_y, 0.0)
        glScalef(self.zoom, self.zoom, 1.0)

        # Wiggle traces 렌더링
        self._render_wiggle_traces()

    def _render_wiggle_traces(self):
        """Wiggle trace 렌더링"""
        trace_end = min(self.trace_start + self.trace_window, self.num_traces)

        for i, trace_idx in enumerate(range(self.trace_start, trace_end)):
            trace_data = self.data[:, trace_idx]
            self._render_single_trace(i, trace_data)

    def _render_single_trace(self, display_idx: int, trace_data: np.ndarray):
        """단일 trace 렌더링"""
        # display_idx: 화면상의 trace 위치 (0부터 시작)
        # trace_data: 실제 trace 데이터

        # Wiggle line (검은색)
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)

        glBegin(GL_LINE_STRIP)
        for sample_idx in range(self.num_samples):
            value = trace_data[sample_idx] * self.data_scale
            x = display_idx + value * self.wiggle_amplitude
            y = sample_idx
            glVertex2f(x, y)
        glEnd()

        # Variable area (양수 부분, 반투명 검은색)
        glColor4f(0.0, 0.0, 0.0, 0.3)

        glBegin(GL_TRIANGLES)
        for sample_idx in range(self.num_samples - 1):
            value1 = trace_data[sample_idx] * self.data_scale
            value2 = trace_data[sample_idx + 1] * self.data_scale

            if value1 > 0 or value2 > 0:
                x1 = display_idx + max(0, value1) * self.wiggle_amplitude
                x2 = display_idx + max(0, value2) * self.wiggle_amplitude
                y1 = sample_idx
                y2 = sample_idx + 1

                # 삼각형 채우기
                glVertex2f(display_idx, y1)
                glVertex2f(x1, y1)
                glVertex2f(display_idx, y2)

                glVertex2f(display_idx, y2)
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
        glEnd()

    def wheelEvent(self, event):
        """마우스 휠 이벤트 (줌)"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))

        self.update()

    def mousePressEvent(self, event):
        """마우스 버튼 눌림"""
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.last_mouse_pos = event.pos()
            self.is_panning = True

    def mouseMoveEvent(self, event):
        """마우스 이동 (패닝)"""
        if self.is_panning and self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            # 화면 좌표계를 데이터 좌표계로 변환
            self.offset_x += dx * 0.1 / self.zoom
            self.offset_y += dy * 0.1 / self.zoom

            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """마우스 버튼 릴리즈"""
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.last_mouse_pos = None

    def keyPressEvent(self, event):
        """키보드 이벤트"""
        step = max(1, self.trace_window // 10)  # 10% 씩 이동

        if event.key() == Qt.Key_Left:
            # 왼쪽으로 이동
            self.trace_start = max(0, self.trace_start - step)
            self.update()
            print(f"[Viewer] Trace range: {self.trace_start} - {self.trace_start + self.trace_window}")

        elif event.key() == Qt.Key_Right:
            # 오른쪽으로 이동
            max_start = max(0, self.num_traces - self.trace_window)
            self.trace_start = min(max_start, self.trace_start + step)
            self.update()
            print(f"[Viewer] Trace range: {self.trace_start} - {self.trace_start + self.trace_window}")

        elif event.key() == Qt.Key_Home:
            # 처음으로
            self.trace_start = 0
            self.update()
            print(f"[Viewer] Trace range: {self.trace_start} - {self.trace_start + self.trace_window}")

        elif event.key() == Qt.Key_End:
            # 끝으로
            self.trace_start = max(0, self.num_traces - self.trace_window)
            self.update()
            print(f"[Viewer] Trace range: {self.trace_start} - {self.trace_start + self.trace_window}")

        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Trace window 줄이기 (더 많은 detail)
            self.trace_window = max(10, self.trace_window - 10)
            self.trace_start = min(self.trace_start, max(0, self.num_traces - self.trace_window))
            self.update()
            print(f"[Viewer] Window size: {self.trace_window} traces")

        elif event.key() == Qt.Key_Minus:
            # Trace window 늘리기 (더 많은 traces)
            self.trace_window = min(self.num_traces, self.trace_window + 10)
            self.update()
            print(f"[Viewer] Window size: {self.trace_window} traces")

        elif event.key() == Qt.Key_R:
            # 뷰 리셋
            self.reset_view()
            self.update()

    def reset_view(self):
        """뷰 초기화"""
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.trace_start = 0
        self.trace_window = min(100, self.num_traces)
        self.update()
