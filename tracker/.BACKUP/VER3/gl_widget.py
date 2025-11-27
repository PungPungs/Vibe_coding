from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QSurfaceFormat
import OpenGL.GL as gl
import numpy as np

class TrackWidget(QOpenGLWidget):
    trackSelected = pyqtSignal(int)  # Emits index of selected track

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []  # List of numpy arrays (N, 2)
        self.track_colors = []
        self.selected_track_index = -1
        
        # Segment selection: (track_index, point_index)
        self.selection_start = None
        self.selection_end = None
        
        # View parameters
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.last_mouse_pos = QPointF()
        
        # OpenGL buffers
        self.vbos = []
        self.vaos = []
        self.shader_program = None
        
        # Set format
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        self.setFormat(fmt)

    def set_tracks(self, tracks):
        self.tracks = tracks
        # Generate random colors for tracks
        self.track_colors = [np.random.rand(3).astype(np.float32) for _ in tracks]
        self.selected_track_index = -1
        self.selection_start = None
        self.selection_end = None
        
        # Reset view to fit all tracks
        self.fit_to_view()
        
        # Re-upload data to GPU
        self.makeCurrent()
        self.cleanup_gl()
        self.init_buffers()
        self.doneCurrent()
        self.update()

    def fit_to_view(self):
        if not self.tracks:
            return
            
        min_x = min(np.min(t[:, 0]) for t in self.tracks)
        max_x = max(np.max(t[:, 0]) for t in self.tracks)
        min_y = min(np.min(t[:, 1]) for t in self.tracks)
        max_y = max(np.max(t[:, 1]) for t in self.tracks)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 and height == 0:
            self.scale = 1.0
        else:
            # Add some padding
            width *= 1.1
            height *= 1.1
            
            if width > height:
                self.scale = 2.0 / width if width > 0 else 1.0
            else:
                self.scale = 2.0 / height if height > 0 else 1.0
                
        self.offset = QPointF(-center_x, -center_y)

    def initializeGL(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glPointSize(10.0) # For markers
        
        # Simple shader
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        uniform vec2 offset;
        uniform float scale;
        uniform float aspectRatio;
        
        void main() {
            vec2 pos = (aPos + offset) * scale;
            // Correct for aspect ratio to keep things square
            if (aspectRatio > 1.0) {
                pos.x /= aspectRatio;
            } else {
                pos.y *= aspectRatio;
            }
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        uniform vec3 color;
        
        void main() {
            FragColor = vec4(color, 1.0);
        }
        """
        
        self.shader_program = self.create_shader_program(vertex_shader, fragment_shader)
        self.init_buffers()

    def create_shader_program(self, vertex_source, fragment_source):
        program = gl.glCreateProgram()
        
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, vertex_source)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            print(gl.glGetShaderInfoLog(vs))
            
        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, fragment_source)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            print(gl.glGetShaderInfoLog(fs))
            
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        
        return program

    def init_buffers(self):
        if not self.tracks:
            return
            
        for track in self.tracks:
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            
            data = track.astype(np.float32).tobytes()
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data, gl.GL_STATIC_DRAW)
            
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, None)
            gl.glEnableVertexAttribArray(0)
            
            self.vaos.append(vao)
            self.vbos.append(vbo)
            
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def cleanup_gl(self):
        if self.vaos:
            gl.glDeleteVertexArrays(len(self.vaos), self.vaos)
            self.vaos = []
        if self.vbos:
            gl.glDeleteBuffers(len(self.vbos), self.vbos)
            self.vbos = []

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        if not self.shader_program or not self.vaos:
            return
            
        gl.glUseProgram(self.shader_program)
        
        # Uniforms
        offset_loc = gl.glGetUniformLocation(self.shader_program, "offset")
        scale_loc = gl.glGetUniformLocation(self.shader_program, "scale")
        aspect_loc = gl.glGetUniformLocation(self.shader_program, "aspectRatio")
        color_loc = gl.glGetUniformLocation(self.shader_program, "color")
        
        gl.glUniform2f(offset_loc, self.offset.x(), self.offset.y())
        gl.glUniform1f(scale_loc, self.scale)
        gl.glUniform1f(aspect_loc, self.width() / self.height())
        
        # Draw tracks
        for i, vao in enumerate(self.vaos):
            if i == self.selected_track_index:
                gl.glLineWidth(3.0)
                gl.glUniform3f(color_loc, 1.0, 0.0, 0.0) # Red for selected
            else:
                gl.glLineWidth(1.0)
                c = self.track_colors[i]
                gl.glUniform3f(color_loc, c[0], c[1], c[2])
                
            gl.glBindVertexArray(vao)
            gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(self.tracks[i]))
            
        # Draw selected segment
        if self.selection_start and self.selection_end:
            t_idx1, p_idx1 = self.selection_start
            t_idx2, p_idx2 = self.selection_end
            
            if t_idx1 == t_idx2 and t_idx1 == self.selected_track_index:
                start = min(p_idx1, p_idx2)
                end = max(p_idx1, p_idx2)
                count = end - start + 1
                
                gl.glLineWidth(5.0)
                gl.glUniform3f(color_loc, 1.0, 1.0, 0.0) # Yellow
                
                gl.glBindVertexArray(self.vaos[t_idx1])
                gl.glDrawArrays(gl.GL_LINE_STRIP, start, count)

        # Draw markers
        if self.selection_start:
            t_idx, p_idx = self.selection_start
            if t_idx == self.selected_track_index:
                gl.glUniform3f(color_loc, 0.0, 1.0, 0.0) # Green
                gl.glBindVertexArray(self.vaos[t_idx])
                gl.glDrawArrays(gl.GL_POINTS, p_idx, 1)
                
        if self.selection_end:
            t_idx, p_idx = self.selection_end
            if t_idx == self.selected_track_index:
                gl.glUniform3f(color_loc, 1.0, 0.0, 1.0) # Magenta
                gl.glBindVertexArray(self.vaos[t_idx])
                gl.glDrawArrays(gl.GL_POINTS, p_idx, 1)

        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.scale *= zoom_factor
        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()
        
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                # Point selection
                self.select_point_at(event.position())
            else:
                # Track selection
                self.select_track_at(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton or event.buttons() & Qt.MouseButton.MiddleButton:
            # Pan
            delta = event.position() - self.last_mouse_pos
            
            dx_ndc = delta.x() / self.width() * 2.0
            dy_ndc = -delta.y() / self.height() * 2.0
            
            aspect = self.width() / self.height()
            
            if aspect > 1.0:
                self.offset.setX(self.offset.x() + dx_ndc * aspect / self.scale)
                self.offset.setY(self.offset.y() + dy_ndc / self.scale)
            else:
                self.offset.setX(self.offset.x() + dx_ndc / self.scale)
                self.offset.setY(self.offset.y() + dy_ndc / aspect / self.scale)

            self.update()
            
        self.last_mouse_pos = event.position()

    def get_world_pos(self, screen_pos):
        x_ndc = (screen_pos.x() / self.width()) * 2.0 - 1.0
        y_ndc = 1.0 - (screen_pos.y() / self.height()) * 2.0
        
        aspect = self.width() / self.height()
        
        if aspect > 1.0:
            world_x = x_ndc * aspect / self.scale - self.offset.x()
            world_y = y_ndc / self.scale - self.offset.y()
        else:
            world_x = x_ndc / self.scale - self.offset.x()
            world_y = y_ndc / aspect / self.scale - self.offset.y()
            
        return np.array([world_x, world_y])

    def select_track_at(self, pos):
        click_point = self.get_world_pos(pos)
        
        min_dist = float('inf')
        closest_idx = -1
        
        threshold_pixels = 10.0
        threshold_ndc = threshold_pixels / self.width() * 2.0
        threshold_world = threshold_ndc / self.scale
        
        for i, track in enumerate(self.tracks):
            dists = np.linalg.norm(track - click_point, axis=1)
            dist = np.min(dists)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        if min_dist < threshold_world:
            self.selected_track_index = closest_idx
            self.trackSelected.emit(closest_idx)
            self.update()
        else:
            if self.selected_track_index != -1:
                self.selected_track_index = -1
                self.trackSelected.emit(-1)
                self.update()

    def select_point_at(self, pos):
        if self.selected_track_index == -1:
            return
            
        click_point = self.get_world_pos(pos)
        track = self.tracks[self.selected_track_index]
        
        dists = np.linalg.norm(track - click_point, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        threshold_pixels = 15.0
        threshold_ndc = threshold_pixels / self.width() * 2.0
        threshold_world = threshold_ndc / self.scale
        
        if min_dist < threshold_world:
            # Determine if this is start or end
            # Logic: if start is not set, set start.
            # If start is set, set end.
            # If both set, reset start to new point, clear end.
            
            if self.selection_start is None:
                self.selection_start = (self.selected_track_index, min_idx)
            elif self.selection_end is None:
                self.selection_end = (self.selected_track_index, min_idx)
            else:
                self.selection_start = (self.selected_track_index, min_idx)
                self.selection_end = None
                
            self.update()

    def get_selected_segment(self):
        if self.selection_start and self.selection_end:
            t1, p1 = self.selection_start
            t2, p2 = self.selection_end
            
            if t1 == t2:
                return t1, min(p1, p2), max(p1, p2)
        return None

