"""
SEG-Y 2D Viewer with First Break Picking
메인 애플리케이션
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QFileDialog, QLabel,
                              QStatusBar, QToolBar, QAction, QMessageBox,
                              QCheckBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from segy_reader import SegyReader
from gl_widget import SegyGLWidget
from picking_manager import PickingManager


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""

    def __init__(self):
        super().__init__()

        # 데이터 관리
        self.segy_reader = SegyReader()
        self.picking_manager = PickingManager()

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('SEG-Y 2D Viewer with First Break Picking')
        self.setGeometry(100, 100, 1200, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # OpenGL 뷰어 위젯
        self.gl_widget = SegyGLWidget()
        self.gl_widget.set_picking_manager(self.picking_manager)
        self.gl_widget.mouse_position_changed.connect(self.on_mouse_position_changed)
        main_layout.addWidget(self.gl_widget)

        # 컨트롤 패널
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # 툴바 생성
        self.create_toolbar()

        # 상태바 생성
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('Ready')
        self.status_bar.addWidget(self.status_label)

        # 마우스 위치 라벨
        self.mouse_pos_label = QLabel('Trace: --, Sample: --')
        self.status_bar.addPermanentWidget(self.mouse_pos_label)

    def create_toolbar(self):
        """툴바 생성"""
        toolbar = QToolBar('Main Toolbar')
        self.addToolBar(toolbar)

        # 파일 열기
        open_action = QAction('Open SEG-Y', self)
        open_action.triggered.connect(self.open_segy_file)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # 뷰 리셋
        reset_action = QAction('Reset View', self)
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()

        # 피킹 저장
        save_picks_action = QAction('Save Picks', self)
        save_picks_action.triggered.connect(self.save_picks)
        toolbar.addAction(save_picks_action)

        # 피킹 로드
        load_picks_action = QAction('Load Picks', self)
        load_picks_action.triggered.connect(self.load_picks)
        toolbar.addAction(load_picks_action)

        # 피킹 클리어
        clear_picks_action = QAction('Clear Picks', self)
        clear_picks_action.triggered.connect(self.clear_picks)
        toolbar.addAction(clear_picks_action)

    def create_control_panel(self) -> QWidget:
        """컨트롤 패널 생성"""
        panel = QWidget()
        layout = QHBoxLayout()
        panel.setLayout(layout)

        # 파일 정보
        self.file_info_label = QLabel('No file loaded')
        layout.addWidget(self.file_info_label)

        layout.addStretch()

        # 피킹 활성화 체크박스
        self.picking_enabled_checkbox = QCheckBox('Enable Picking')
        self.picking_enabled_checkbox.setChecked(True)
        self.picking_enabled_checkbox.stateChanged.connect(self.on_picking_enabled_changed)
        layout.addWidget(self.picking_enabled_checkbox)

        # 피킹 표시 체크박스
        self.show_picks_checkbox = QCheckBox('Show Picks')
        self.show_picks_checkbox.setChecked(True)
        self.show_picks_checkbox.stateChanged.connect(self.on_show_picks_changed)
        layout.addWidget(self.show_picks_checkbox)

        # 컬러맵 선택
        layout.addWidget(QLabel('Colormap:'))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['seismic', 'grayscale'])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        return panel

    def open_segy_file(self):
        """SEG-Y 파일 열기"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Open SEG-Y File',
            '',
            'SEG-Y Files (*.sgy *.segy);;All Files (*)'
        )

        if filename:
            self.status_label.setText('Loading SEG-Y file...')
            QApplication.processEvents()

            success = self.segy_reader.load_file(filename)

            if success:
                # 데이터 가져오기
                data = self.segy_reader.get_data()
                num_traces, num_samples = self.segy_reader.get_dimensions()

                # 피킹 매니저 초기화
                self.picking_manager.set_num_traces(num_traces)
                self.picking_manager.clear_picks()

                # GL 위젯에 데이터 설정
                self.gl_widget.set_data(data)

                # 파일 정보 업데이트
                sample_rate = self.segy_reader.get_sample_rate()
                self.file_info_label.setText(
                    f'File: {filename} | Traces: {num_traces} | '
                    f'Samples: {num_samples} | Sample Rate: {sample_rate*1000:.2f} ms'
                )

                self.status_label.setText('SEG-Y file loaded successfully')
            else:
                QMessageBox.critical(self, 'Error', 'Failed to load SEG-Y file')
                self.status_label.setText('Failed to load file')

    def reset_view(self):
        """뷰 리셋"""
        self.gl_widget.reset_view()
        self.status_label.setText('View reset')

    def save_picks(self):
        """피킹 저장"""
        if len(self.picking_manager.get_picks()) == 0:
            QMessageBox.warning(self, 'Warning', 'No picks to save')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Picks',
            '',
            'CSV Files (*.csv);;All Files (*)'
        )

        if filename:
            success = self.picking_manager.save_to_file(filename)
            if success:
                self.status_label.setText(f'Picks saved to {filename}')
                QMessageBox.information(self, 'Success', 'Picks saved successfully')
            else:
                QMessageBox.critical(self, 'Error', 'Failed to save picks')

    def load_picks(self):
        """피킹 로드"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Picks',
            '',
            'CSV Files (*.csv);;All Files (*)'
        )

        if filename:
            success = self.picking_manager.load_from_file(filename)
            if success:
                self.status_label.setText(f'Picks loaded from {filename}')
                QMessageBox.information(self, 'Success', 'Picks loaded successfully')
            else:
                QMessageBox.critical(self, 'Error', 'Failed to load picks')

    def clear_picks(self):
        """피킹 클리어"""
        reply = QMessageBox.question(
            self,
            'Confirm',
            'Are you sure you want to clear all picks?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.picking_manager.clear_picks()
            self.status_label.setText('Picks cleared')

    def on_picking_enabled_changed(self, state):
        """피킹 활성화 상태 변경"""
        enabled = state == Qt.Checked
        self.picking_manager.set_picking_enabled(enabled)
        self.status_label.setText(f'Picking {"enabled" if enabled else "disabled"}')

    def on_show_picks_changed(self, state):
        """피킹 표시 상태 변경"""
        show = state == Qt.Checked
        self.gl_widget.set_show_picks(show)

    def on_colormap_changed(self, colormap: str):
        """컬러맵 변경"""
        self.gl_widget.set_colormap(colormap)
        self.status_label.setText(f'Colormap changed to {colormap}')

    def on_mouse_position_changed(self, trace_idx: int, sample_idx: float):
        """마우스 위치 변경"""
        self.mouse_pos_label.setText(f'Trace: {trace_idx}, Sample: {sample_idx:.2f}')

    def closeEvent(self, event):
        """윈도우 종료 이벤트"""
        self.segy_reader.close()
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setApplicationName('SEG-Y 2D Viewer')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
