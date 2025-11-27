from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QPushButton, QFileDialog, QSplitter, QLabel,
                             QInputDialog, QMessageBox, QTextEdit, QMenu)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QAction
import os
import numpy as np
from segy_reader import SegyReader
from gl_widget import TrackWidget
from trace_viewer import SegyTraceViewer

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

        # Clear Button
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        left_layout.addWidget(self.btn_clear)
        
        # File List
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.currentRowChanged.connect(self.on_list_selection_changed)
        # Enable Context Menu
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.open_context_menu)
        left_layout.addWidget(self.file_list)
        
        # Info Panel
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Select a track to see info...")
        left_layout.addWidget(self.info_text)
        
        # Info Label (Status)
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
        
        # Keep reference to viewer to prevent garbage collection
        self.trace_viewer = None

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
        self.info_text.clear()
        
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

    def clear_all(self):
        self.loaded_files = []
        self.tracks = []
        self.readers = []
        self.file_list.clear()
        self.info_text.clear()
        self.gl_widget.set_tracks([])
        self.info_label.setText("No files loaded")
        self.btn_split.setEnabled(False)
        self.btn_extract.setEnabled(False)
        if self.trace_viewer:
            self.trace_viewer.close()
            self.trace_viewer = None

    def on_list_selection_changed(self, row):
        if row >= 0:
            # Only update GL if changed (prevents double update when clicking in GL)
            if self.gl_widget.selected_track_index != row:
                self.gl_widget.selected_track_index = row
                self.gl_widget.update()
            
            self.btn_split.setEnabled(True)
            self.btn_extract.setEnabled(True)
            
            # Update Info Panel
            track = self.tracks[row]
            filename = os.path.basename(self.loaded_files[row])
            trace_count = len(track)
            
            # Calculate length
            dists = np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1))
            total_length = np.sum(dists)
            
            min_x = np.min(track[:, 0])
            max_x = np.max(track[:, 0])
            min_y = np.min(track[:, 1])
            max_y = np.max(track[:, 1])
            
            info = f"<b>File:</b> {filename}<br>"
            info += f"<b>Traces:</b> {trace_count}<br>"
            info += f"<b>Length:</b> {total_length:.2f} units<br>"
            info += f"<b>X Range:</b> {min_x:.2f} - {max_x:.2f}<br>"
            info += f"<b>Y Range:</b> {min_y:.2f} - {max_y:.2f}"
            
            self.info_text.setHtml(info)
            
        else:
            if self.gl_widget.selected_track_index != -1:
                self.gl_widget.selected_track_index = -1
                self.gl_widget.update()
                
            self.btn_split.setEnabled(False)
            self.btn_extract.setEnabled(False)
            self.info_text.clear()

    def on_gl_track_selected(self, index):
        # We don't block signals here anymore, so that setCurrentRow triggers
        # on_list_selection_changed, which updates the Info Panel and buttons.
        if index >= 0:
            self.file_list.setCurrentRow(index)
        else:
            self.file_list.clearSelection()

    def open_context_menu(self, position: QPoint):
        item = self.file_list.itemAt(position)
        if item:
            menu = QMenu()
            action_view = QAction("Open Trace Viewer", self)
            action_view.triggered.connect(self.open_trace_viewer)
            menu.addAction(action_view)
            menu.exec(self.file_list.mapToGlobal(position))

    def open_trace_viewer(self):
        row = self.file_list.currentRow()
        if row >= 0:
            filepath = self.loaded_files[row]
            # Close existing if open
            if self.trace_viewer:
                self.trace_viewer.close()
            
            self.trace_viewer = SegyTraceViewer(filepath)
            self.trace_viewer.show()

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
