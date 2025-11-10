"""
SEG-Y Trace Viewer
Simple viewer with zoom and horizontal navigation
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QPushButton, QFileDialog, QLabel, QStatusBar,
                              QToolBar, QAction, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt

from segy_loader import SegyLoader
from gl_viewer import TraceViewer


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()

        # SEG-Y 로더
        self.loader = SegyLoader()

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('SEG-Y Trace Viewer')
        self.setGeometry(100, 100, 1400, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # OpenGL Viewer
        self.viewer = TraceViewer()
        layout.addWidget(self.viewer)

        # 컨트롤 힌트
        hint_layout = QHBoxLayout()
        hint_label = QLabel(
            'Controls: Mouse Wheel (Zoom) | Right-Click Drag (Pan) | '
            'Left/Right Arrow (Navigate) | +/- (Window Size) | R (Reset) | Home/End'
        )
        hint_label.setStyleSheet('padding: 5px; background-color: #f0f0f0;')
        hint_layout.addWidget(hint_label)
        layout.addLayout(hint_layout)

        # 툴바 생성
        self.create_toolbar()

        # 상태바
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('Ready')
        self.status_bar.addWidget(self.status_label)

        # 정보 라벨
        self.info_label = QLabel('')
        self.status_bar.addPermanentWidget(self.info_label)

    def create_toolbar(self):
        """툴바 생성"""
        toolbar = QToolBar('Main Toolbar')
        self.addToolBar(toolbar)

        # 파일 열기
        open_action = QAction('Open SEG-Y', self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # 뷰 리셋
        reset_action = QAction('Reset View', self)
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()

        # 도움말
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

    def open_file(self):
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

            success = self.loader.load(filename)

            if success:
                # 데이터 가져오기
                data = self.loader.get_data()
                num_samples, num_traces = self.loader.get_dimensions()
                sample_rate = self.loader.get_sample_rate()

                # Viewer에 데이터 설정
                self.viewer.set_data(data)

                # 정보 업데이트
                self.info_label.setText(
                    f'Traces: {num_traces} | Samples: {num_samples} | '
                    f'Sample Rate: {sample_rate * 1000:.2f} ms'
                )

                self.status_label.setText(f'Loaded: {filename}')
            else:
                QMessageBox.critical(self, 'Error', 'Failed to load SEG-Y file')
                self.status_label.setText('Failed to load file')

    def reset_view(self):
        """뷰 리셋"""
        self.viewer.reset_view()
        self.status_label.setText('View reset')

    def show_help(self):
        """도움말 표시"""
        help_text = """
<h3>SEG-Y Trace Viewer - Controls</h3>

<p><b>File Operations:</b></p>
<ul>
<li>Open SEG-Y: Load a SEG-Y file</li>
<li>Reset View: Reset zoom and position</li>
</ul>

<p><b>Mouse Controls:</b></p>
<ul>
<li>Mouse Wheel: Zoom in/out</li>
<li>Right-Click Drag: Pan the view</li>
</ul>

<p><b>Keyboard Controls:</b></p>
<ul>
<li><b>Left Arrow</b>: Move to previous traces</li>
<li><b>Right Arrow</b>: Move to next traces</li>
<li><b>Home</b>: Jump to first traces</li>
<li><b>End</b>: Jump to last traces</li>
<li><b>+/=</b>: Decrease window size (show fewer traces, more detail)</li>
<li><b>-</b>: Increase window size (show more traces, less detail)</li>
<li><b>R</b>: Reset view</li>
</ul>

<p><b>Tips:</b></p>
<ul>
<li>Use arrow keys to navigate through large datasets</li>
<li>Adjust window size with +/- for optimal viewing</li>
<li>Combine zoom and pan for detailed inspection</li>
</ul>
        """
        QMessageBox.information(self, 'Help', help_text)

    def closeEvent(self, event):
        """윈도우 종료"""
        self.loader.close()
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setApplicationName('SEG-Y Trace Viewer')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
