"""
SEG-Y 2D Viewer with First Break Picking
메인 애플리케이션
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QFileDialog, QLabel,
                              QStatusBar, QToolBar, QAction, QMessageBox,
                              QCheckBox, QComboBox, QSlider, QLineEdit)
from PyQt5.QtGui import QIcon, QDoubleValidator

from segy_reader import SegyReader
from gl_widget import SegyGLWidget
from picking_manager import PickingManager
from auto_picking import AutoPicker, get_algorithm_params


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""

    def __init__(self):
        super().__init__()

        # 데이터 관리
        self.segy_reader = SegyReader()
        self.picking_manager = PickingManager()
        self.auto_picker = AutoPicker()

        # 시간 자르기 변수
        self.time_start_ms: float = 0.0
        self.time_end_ms: float = 0.0

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('SEG-Y 2D Viewer with First Break Picking')
        self.setGeometry(50, 50, 1600, 900)  # 더 큰 윈도우

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃 (가로)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # OpenGL 뷰어 위젯 (왼쪽, 넓게)
        self.gl_widget = SegyGLWidget()
        self.gl_widget.set_picking_manager(self.picking_manager)
        self.gl_widget.mouse_position_changed.connect(self.on_mouse_position_changed)
        self.gl_widget.setMinimumWidth(1000)  # 최소 너비 설정
        main_layout.addWidget(self.gl_widget, stretch=4)  # 4:1 비율

        # 컨트롤 패널 (오른쪽)
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(300)  # 최대 너비 제한
        main_layout.addWidget(control_panel, stretch=1)

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

        toolbar.addSeparator()

        # 자동 피킹
        auto_pick_action = QAction('Auto Pick', self)
        auto_pick_action.triggered.connect(self.auto_pick)
        toolbar.addAction(auto_pick_action)

    def create_control_panel(self) -> QWidget:
        """컨트롤 패널 생성 (오른쪽 세로)"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 타이틀
        title_label = QLabel('Control Panel')
        title_label.setStyleSheet('font-weight: bold; font-size: 14px; padding: 5px;')
        layout.addWidget(title_label)

        layout.addSpacing(10)

        # 파일 정보 그룹
        file_group_label = QLabel('File Information')
        file_group_label.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(file_group_label)

        self.file_info_label = QLabel('No file loaded')
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet('padding: 5px; background-color: #f0f0f0; border-radius: 3px;')
        layout.addWidget(self.file_info_label)

        layout.addSpacing(15)

        # 피킹 옵션 그룹
        picking_group_label = QLabel('Picking Options')
        picking_group_label.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(picking_group_label)

        # 피킹 활성화 체크박스
        self.picking_enabled_checkbox = QCheckBox('Enable Manual Picking')
        self.picking_enabled_checkbox.setChecked(True)
        self.picking_enabled_checkbox.stateChanged.connect(self.on_picking_enabled_changed)
        layout.addWidget(self.picking_enabled_checkbox)

        # 피킹 표시 체크박스
        self.show_picks_checkbox = QCheckBox('Show Picks')
        self.show_picks_checkbox.setChecked(True)
        self.show_picks_checkbox.stateChanged.connect(self.on_show_picks_changed)
        layout.addWidget(self.show_picks_checkbox)

        layout.addSpacing(15)

        # 디스플레이 옵션 그룹
        display_group_label = QLabel('Display Options')
        display_group_label.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(display_group_label)

        # Wiggle 표시 체크박스
        self.show_wiggle_checkbox = QCheckBox('Show Wiggle')
        self.show_wiggle_checkbox.setChecked(True)
        self.show_wiggle_checkbox.stateChanged.connect(self.on_show_wiggle_changed)
        layout.addWidget(self.show_wiggle_checkbox)

        # Variable Area 표시 체크박스
        self.show_va_checkbox = QCheckBox('Show Variable Area')
        self.show_va_checkbox.setChecked(True)
        self.show_va_checkbox.stateChanged.connect(self.on_show_va_changed)
        layout.addWidget(self.show_va_checkbox)

        # Wiggle 스케일 슬라이더
        wiggle_scale_label = QLabel('Wiggle Scale:')
        layout.addWidget(wiggle_scale_label)
        self.wiggle_scale_slider = QSlider(Qt.Horizontal)
        self.wiggle_scale_slider.setMinimum(10)
        self.wiggle_scale_slider.setMaximum(200)
        self.wiggle_scale_slider.setValue(100)
        self.wiggle_scale_slider.valueChanged.connect(self.on_wiggle_scale_changed)
        layout.addWidget(self.wiggle_scale_slider)

        layout.addSpacing(15)

        # 시간 구간 자르기 그룹
        cropping_group_label = QLabel('Time Cropping (ms)')
        cropping_group_label.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(cropping_group_label)

        # 시작 시간
        time_start_layout = QHBoxLayout()
        time_start_layout.addWidget(QLabel('Start:'))
        self.time_start_input = QLineEdit('0.0')
        self.time_start_input.setValidator(QDoubleValidator())
        self.time_start_input.editingFinished.connect(self.on_time_cropping_changed)
        time_start_layout.addWidget(self.time_start_input)
        layout.addLayout(time_start_layout)

        # 종료 시간
        time_end_layout = QHBoxLayout()
        time_end_layout.addWidget(QLabel('End:'))
        self.time_end_input = QLineEdit('0.0') # Will be updated to max time on load
        self.time_end_input.setValidator(QDoubleValidator())
        self.time_end_input.editingFinished.connect(self.on_time_cropping_changed)
        time_end_layout.addWidget(self.time_end_input)
        layout.addLayout(time_end_layout)

        layout.addSpacing(15)

        # 자동 피킹 그룹
        auto_pick_group_label = QLabel('Auto Picking')
        auto_pick_group_label.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(auto_pick_group_label)

        # 자동 피킹 알고리즘 선택
        auto_pick_label = QLabel('Algorithm:')
        layout.addWidget(auto_pick_label)
        self.auto_pick_algo_combo = QComboBox()
        self.auto_pick_algo_combo.addItems(['STA/LTA', 'Energy Ratio', 'AIC'])
        layout.addWidget(self.auto_pick_algo_combo)

        # 자동 피킹 버튼
        auto_pick_btn = QPushButton('Run Auto Pick')
        auto_pick_btn.clicked.connect(self.auto_pick)
        auto_pick_btn.setStyleSheet('padding: 8px; background-color: #4CAF50; color: white; font-weight: bold;')
        layout.addWidget(auto_pick_btn)

        layout.addStretch()

        # 도움말
        help_label = QLabel('Controls:\n\n'
                           '• Left Click: Pick\n'
                           '• Mouse Wheel: Zoom\n'
                           '• Right/Middle Drag: Pan')
        help_label.setStyleSheet('padding: 10px; background-color: #f9f9f9; border-radius: 3px; font-size: 10px;')
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

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
                # 데이터 가져오기 (num_samples x num_traces)
                data = self.segy_reader.get_data()
                num_samples, num_traces = self.segy_reader.get_dimensions()

                # 피킹 매니저 초기화
                self.picking_manager.set_num_traces(num_traces)
                self.picking_manager.clear_picks()

                # GL 위젯에 데이터 설정
                self.gl_widget.set_data(data)
                self.gl_widget.sample_rate = sample_rate # Pass sample_rate

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

    def on_show_wiggle_changed(self, state):
        """Wiggle 표시 상태 변경"""
        show = state == Qt.Checked
        self.gl_widget.set_show_wiggle(show)
        self.status_label.setText(f'Wiggle {"shown" if show else "hidden"}')

    def on_show_va_changed(self, state):
        """Variable Area 표시 상태 변경"""
        show = state == Qt.Checked
        self.gl_widget.set_show_va(show)
        self.status_label.setText(f'Variable Area {"shown" if show else "hidden"}')

    def on_wiggle_scale_changed(self, value):
        """Wiggle 스케일 변경"""
        scale = value / 100.0  # 0.1 ~ 2.0
        self.gl_widget.set_wiggle_scale(scale)
        self.status_label.setText(f'Wiggle scale: {scale:.2f}')

    def auto_pick(self):
        """자동 피킹 실행"""
        # 데이터가 로드되었는지 확인
        raw_data = self.segy_reader.get_raw_data()
        if raw_data is None:
            QMessageBox.warning(self, 'Warning', 'No SEG-Y file loaded')
            return

        # 알고리즘 선택
        algo_name = self.auto_pick_algo_combo.currentText()
        algo_map = {
            'STA/LTA': 'sta_lta',
            'Energy Ratio': 'energy_ratio',
            'AIC': 'aic'
        }
        algorithm = algo_map.get(algo_name, 'sta_lta')

        # 상태 메시지
        self.status_label.setText(f'Running auto picking with {algo_name}...')
        QApplication.processEvents()

        try:
            # 알고리즘 파라미터 가져오기
            params = get_algorithm_params(algorithm)
            params['use_parallel'] = True

            # 자동 피킹 실행
            picks = self.auto_picker.pick_all_traces(raw_data, algorithm, **params)

            # 피킹 결과를 매니저에 추가
            self.picking_manager.clear_picks()
            for trace_idx, sample_idx in picks:
                self.picking_manager.add_pick(trace_idx, sample_idx)

            # 결과 메시지
            self.status_label.setText(
                f'Auto picking completed: {len(picks)} picks ({algo_name})'
            )

            if len(picks) == 0:
                QMessageBox.information(
                    self,
                    'Auto Picking',
                    'No first breaks detected. Try adjusting the algorithm or parameters.'
                )
            else:
                QMessageBox.information(
                    self,
                    'Auto Picking',
                    f'Successfully picked {len(picks)} first breaks using {algo_name}.'
                )

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Auto picking failed: {str(e)}')
            self.status_label.setText('Auto picking failed')
            import traceback
            traceback.print_exc()

    def on_mouse_position_changed(self, trace_idx: int, sample_idx: float):
        """마우스 위치 변경"""
        self.mouse_pos_label.setText(f'Trace: {trace_idx}, Sample: {sample_idx:.2f}')

    def on_time_cropping_changed(self):
        """시간 자르기 입력 변경 시 호출"""
        try:
            self.time_start_ms = float(self.time_start_input.text())
            self.time_end_ms = float(self.time_end_input.text())
            self.gl_widget.set_time_cropping(self.time_start_ms, self.time_end_ms)
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers for time cropping.')

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
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
