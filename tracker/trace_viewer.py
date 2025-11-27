import numpy as np
import segyio
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QRadioButton, QButtonGroup, QLabel, QSlider, QSpinBox,
                             QGroupBox, QToolBar)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QBrush

class TraceView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(0) # Disable antialiasing for performance
        self.picks = [] # List of (trace_idx, sample_idx)
        self.pick_items = []
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None

    def set_image(self, image):
        self.scene.clear()
        self.picks = []
        self.pick_items = []
        
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Map to scene coordinates
            pos = self.mapToScene(event.pos())
            x = pos.x()
            y = pos.y()
            
            # Check bounds
            if self.pixmap_item:
                rect = self.pixmap_item.boundingRect()
                if rect.contains(x, y):
                    self.add_pick(x, y)
                    
        super().mousePressEvent(event)

    def add_pick(self, x, y):
        # Add a visual marker
        r = 2
        dot = self.scene.addEllipse(x - r, y - r, r * 2, r * 2, 
                                    QPen(Qt.GlobalColor.red), QBrush(Qt.GlobalColor.red))
        self.picks.append((x, y))
        self.pick_items.append(dot)

class SegyTraceViewer(QMainWindow):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.setWindowTitle(f"Trace Viewer - {filepath}")
        self.resize(1000, 600)
        
        self.data = None
        self.processed_data = None
        
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Controls Area
        controls = QHBoxLayout()
        
        # Normalization Group
        norm_group = QGroupBox("Normalization")
        norm_layout = QHBoxLayout()
        self.radio_raw = QRadioButton("Raw")
        self.radio_rms = QRadioButton("RMS")
        self.radio_max = QRadioButton("Max")
        self.radio_max.setChecked(True)
        
        self.norm_btn_group = QButtonGroup()
        self.norm_btn_group.addButton(self.radio_raw)
        self.norm_btn_group.addButton(self.radio_rms)
        self.norm_btn_group.addButton(self.radio_max)
        
        self.norm_btn_group.buttonClicked.connect(self.update_view)
        
        norm_layout.addWidget(self.radio_raw)
        norm_layout.addWidget(self.radio_rms)
        norm_layout.addWidget(self.radio_max)
        norm_group.setLayout(norm_layout)
        controls.addWidget(norm_group)
        
        # Gain Control
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(1, 100)
        self.slider_gain.setValue(10)
        self.slider_gain.valueChanged.connect(self.update_view)
        gain_layout.addWidget(self.slider_gain)
        controls.addLayout(gain_layout)
        
        # Bandwidth (Placeholder)
        bw_group = QGroupBox("Bandwidth (Hz)")
        bw_layout = QHBoxLayout()
        bw_layout.addWidget(QLabel("Low:"))
        self.spin_low = QSpinBox()
        self.spin_low.setRange(0, 500)
        self.spin_low.setValue(0)
        bw_layout.addWidget(self.spin_low)
        
        bw_layout.addWidget(QLabel("High:"))
        self.spin_high = QSpinBox()
        self.spin_high.setRange(0, 500)
        self.spin_high.setValue(100)
        bw_layout.addWidget(self.spin_high)
        bw_group.setLayout(bw_layout)
        controls.addWidget(bw_group)
        
        layout.addLayout(controls)
        
        # Graphics View
        self.view = TraceView()
        layout.addWidget(self.view)

    def load_data(self):
        try:
            # Try big endian first, then little
            try:
                with segyio.open(self.filepath, ignore_geometry=True, endian='big') as f:
                    self.data = np.stack([t for t in f.trace])
            except:
                with segyio.open(self.filepath, ignore_geometry=True, endian='little') as f:
                    self.data = np.stack([t for t in f.trace])
            
            # Transpose so traces are columns (Time on Y axis)
            self.data = self.data.T
            self.update_view()
            
        except Exception as e:
            print(f"Error loading SEG-Y: {e}")

    def update_view(self):
        if self.data is None:
            return
            
        # 1. Apply Normalization
        temp_data = self.data.copy()
        
        if self.radio_rms.isChecked():
            # Trace-wise RMS normalization
            rms = np.sqrt(np.mean(temp_data**2, axis=0))
            rms[rms == 0] = 1
            temp_data = temp_data / rms
        elif self.radio_max.isChecked():
            # Global Max normalization (simple)
            m = np.max(np.abs(temp_data))
            if m > 0:
                temp_data = temp_data / m
        
        # 2. Apply Gain
        gain = self.slider_gain.value() / 10.0
        temp_data = temp_data * gain
        
        # Clip to -1, 1 for visualization
        temp_data = np.clip(temp_data, -1, 1)
        
        # Convert to 0-255 grayscale
        # -1 -> 0 (Black), 0 -> 127 (Gray), 1 -> 255 (White)
        # Or usually seismic is: Positive=Black/Blue, Negative=Red/White?
        # Let's do simple grayscale: 0=127, +1=255, -1=0
        img_data = ((temp_data + 1) / 2 * 255).astype(np.uint8)
        
        # Create QImage
        height, width = img_data.shape
        # QImage requires C-contiguous data
        if not img_data.flags['C_CONTIGUOUS']:
            img_data = img_data.copy(order='C')
            
        image = QImage(img_data.data, width, height, img_data.strides[0], QImage.Format.Format_Grayscale8)
        self.view.set_image(image)
