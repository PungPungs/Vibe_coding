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
            # Selection logic
            self.select_track_at(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton or event.buttons() & Qt.MouseButton.MiddleButton:
            # Pan
            delta = event.position() - self.last_mouse_pos
            
            # Convert screen delta to world delta
            # Screen width corresponds to 2.0 / scale in world units (roughly, ignoring aspect for now)
            # Actually we need to be careful with aspect ratio.
            
            aspect = self.width() / self.height()
            
            dx = delta.x() / self.width() * 2.0 / self.scale
            dy = -delta.y() / self.height() * 2.0 / self.scale
            
            if aspect > 1.0:
                dx *= aspect
            else:
                dy /= aspect # Wait, if aspect < 1 (tall), y range is larger?
                # Let's look at shader:
                # if aspect > 1: pos.x /= aspect -> x is squeezed. So world x range is larger.
                # if aspect < 1: pos.y *= aspect -> y is squeezed.
            
            # Let's simplify:
            # NDC is -1 to 1.
            # Screen is 0 to w, 0 to h.
            # dx_ndc = dx_screen / w * 2
            # dy_ndc = -dy_screen / h * 2
            
            # pos_ndc = (pos_world + offset) * scale / aspect_correction
            # pos_world = pos_ndc * aspect_correction / scale - offset
            # delta_world = delta_ndc * aspect_correction / scale
            
            dx_ndc = delta.x() / self.width() * 2.0
            dy_ndc = -delta.y() / self.height() * 2.0
            
            if aspect > 1.0:
                dx_ndc *= aspect
            else:
                dy_ndc /= aspect # Correct?
                # Shader: pos.y *= aspect. So pos_ndc.y = pos_world.y * scale * aspect.
                # pos_world.y = pos_ndc.y / (scale * aspect)
                # So dy_world = dy_ndc / (scale * aspect)
            
            # Wait, let's just do it empirically or strictly.
            # Shader:
            # if aspect > 1: x' = (x+ox)*s / aspect, y' = (y+oy)*s
            # if aspect <= 1: x' = (x+ox)*s, y' = (y+oy)*s * aspect
            
            if aspect > 1.0:
                self.offset.setX(self.offset.x() + dx_ndc / self.scale * aspect) # Wait, x' = ... / aspect. So dx' = dx * s / aspect. dx = dx' * aspect / s
                self.offset.setX(self.offset.x() + dx_ndc * aspect / self.scale)
                self.offset.setY(self.offset.y() + dy_ndc / self.scale)
            else:
                self.offset.setX(self.offset.x() + dx_ndc / self.scale)
                self.offset.setY(self.offset.y() + dy_ndc / aspect / self.scale) # y' = ... * aspect. dy = dy' / aspect / s

            self.update()
            
        self.last_mouse_pos = event.position()

    def select_track_at(self, pos):
        # Ray casting / Distance check
        # Convert click pos to world coordinates
        x_ndc = (pos.x() / self.width()) * 2.0 - 1.0
        y_ndc = 1.0 - (pos.y() / self.height()) * 2.0
        
        aspect = self.width() / self.height()
        
        # Inverse of shader transform
        # if aspect > 1: x_ndc = (x+ox)*s / aspect -> x = x_ndc * aspect / s - ox
        # if aspect <= 1: y_ndc = (y+oy)*s * aspect -> y = y_ndc / aspect / s - oy
        
        world_x, world_y = 0, 0
        
        if aspect > 1.0:
            world_x = x_ndc * aspect / self.scale - self.offset.x()
            world_y = y_ndc / self.scale - self.offset.y()
        else:
            world_x = x_ndc / self.scale - self.offset.x()
            world_y = y_ndc / aspect / self.scale - self.offset.y()
            
        click_point = np.array([world_x, world_y])
        
        # Find closest track
        min_dist = float('inf')
        closest_idx = -1
        
        # Threshold for selection (in world units? or screen units?)
        # Screen units is better for UX.
        # Let's compute distance in world units but compare against a threshold that scales with zoom.
        threshold_pixels = 10.0
        threshold_ndc = threshold_pixels / self.width() * 2.0
        threshold_world = threshold_ndc / self.scale # Rough approx
        
        for i, track in enumerate(self.tracks):
            # Simple check: distance to points. For lines, we should check segments, but points might be enough if dense.
            # Let's check distance to nearest point for efficiency first.
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

