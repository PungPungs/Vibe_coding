from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QPushButton, QFileDialog, QSplitter, QLabel,
                             QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt
import os
from segy_reader import SegyReader
from gl_widget import TrackWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEG-Y Track Viewer")
        self.resize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (Controls & List)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Open Button
        self.btn_open = QPushButton("Open SEG-Y Files")
        self.btn_open.clicked.connect(self.open_files)
        left_layout.addWidget(self.btn_open)
        
        # Split Button
        self.btn_split = QPushButton("Split Track")
        self.btn_split.clicked.connect(self.split_current_track)
        self.btn_split.setEnabled(False)
        left_layout.addWidget(self.btn_split)
        
        # Extract Button
        self.btn_extract = QPushButton("Extract Segment")
        self.btn_extract.clicked.connect(self.extract_segment)
        self.btn_extract.setEnabled(False)
        left_layout.addWidget(self.btn_extract)
        
        # File List
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.currentRowChanged.connect(self.on_list_selection_changed)
        left_layout.addWidget(self.file_list)
        
        # Info Label
        self.info_label = QLabel("No files loaded")
        left_layout.addWidget(self.info_label)
        
        splitter.addWidget(left_panel)
        
        # Right panel (OpenGL)
        self.gl_widget = TrackWidget()
        self.gl_widget.trackSelected.connect(self.on_gl_track_selected)
        splitter.addWidget(self.gl_widget)
        
        # Set splitter sizes (20% left, 80% right)
        splitter.setSizes([240, 960])
        
        # Data
        self.loaded_files = [] # List of filenames
        self.tracks = [] # List of numpy arrays
        self.readers = [] # List of SegyReader objects

    def open_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Open SEG-Y Files", "", "SEG-Y Files (*.sgy *.segy)"
        )
        
        if not filenames:
            return
            
        self.loaded_files = []
        self.tracks = []
        self.readers = []
        self.file_list.clear()
        
        for fn in filenames:
            reader = SegyReader(fn)
            coords = reader.read_coords()
            
            if coords is not None and len(coords) > 0:
                self.loaded_files.append(fn)
                self.tracks.append(coords)
                self.readers.append(reader)
                self.file_list.addItem(os.path.basename(fn))
            else:
                print(f"Failed to load or empty: {fn}")
                
        self.gl_widget.set_tracks(self.tracks)
        self.info_label.setText(f"Loaded {len(self.tracks)} files")
        self.btn_split.setEnabled(False)
        self.btn_extract.setEnabled(False)

    def on_list_selection_changed(self, row):
        if row >= 0:
            self.gl_widget.selected_track_index = row
            self.gl_widget.update()
            self.btn_split.setEnabled(True)
            self.btn_extract.setEnabled(True)
        else:
            self.gl_widget.selected_track_index = -1
            self.gl_widget.update()
            self.btn_split.setEnabled(False)
            self.btn_extract.setEnabled(False)

    def on_gl_track_selected(self, index):
        if index >= 0:
            # Block signals to prevent loop
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(index)
            self.file_list.blockSignals(False)
            self.btn_split.setEnabled(True)
            self.btn_extract.setEnabled(True)
        else:
            self.file_list.clearSelection()
            self.btn_split.setEnabled(False)
            self.btn_extract.setEnabled(False)

    def split_current_track(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
            
        reader = self.readers[row]
        
        distance, ok = QInputDialog.getDouble(self, "Split Track", 
                                            "Enter split interval (meters):", 
                                            100.0, 1.0, 100000.0, 1)
        if ok:
            output_dir = os.path.dirname(reader.filepath)
            success, msg = reader.split_file(distance, output_dir)
            
            if success:
                QMessageBox.information(self, "Success", msg)
            else:
                QMessageBox.critical(self, "Error", f"Failed to split file:\n{msg}")

    def extract_segment(self):
        segment = self.gl_widget.get_selected_segment()
        if not segment:
            QMessageBox.warning(self, "Warning", "Please select a segment first using Shift + Click (Start) and Shift + Click (End).")
            return
            
        track_idx, start_idx, end_idx = segment
        
        # Ensure the selected track matches the current file list selection
        # (It should, because on_gl_track_selected updates the list)
        if track_idx != self.file_list.currentRow():
            QMessageBox.warning(self, "Warning", "Selection mismatch.")
            return
            
        reader = self.readers[track_idx]
        
        default_name = f"{os.path.splitext(os.path.basename(reader.filepath))[0]}_segment.sgy"
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Segment", default_name, "SEG-Y Files (*.sgy *.segy)")
        
        if output_path:
            success, msg = reader.extract_segment(start_idx, end_idx, output_path)
            
            if success:
                QMessageBox.information(self, "Success", msg)
            else:
                QMessageBox.critical(self, "Error", f"Failed to extract segment:\n{msg}")
