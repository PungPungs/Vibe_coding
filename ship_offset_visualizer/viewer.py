"""OpenGL viewer for ship and sensor offsets."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import pyglet
    from pyglet.gl import (
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LINES,
        GL_POINTS,
        GL_POLYGON_OFFSET_FILL,
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glDisable,
        glEnable,
        glEnd,
        glLoadIdentity,
        glMatrixMode,
        glPointSize,
        glPolygonOffset,
        glPopMatrix,
        glPushMatrix,
        glRotatef,
        glTranslatef,
        glVertex3f,
        gluLookAt,
        gluPerspective,
    )
    from pyglet.gl import GL_MODELVIEW, GL_PROJECTION
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "pyglet is required for visualization. Install it with `pip install pyglet`."
    ) from exc

from .data import OffsetRecord, SensorOffset
from .inventor import InventorMesh, load_inventor


class OrbitCamera:
    def __init__(self) -> None:
        self.distance = 10.0
        self.azimuth = 45.0
        self.elevation = 30.0

    def apply(self) -> None:
        glTranslatef(0.0, 0.0, -self.distance)
        glRotatef(self.elevation, 1.0, 0.0, 0.0)
        glRotatef(self.azimuth, 0.0, 1.0, 0.0)


class OffsetViewer(pyglet.window.Window):
    """Interactive viewer displaying ship geometry with sensor offsets."""

    def __init__(
        self,
        record: OffsetRecord,
        mesh: Optional[InventorMesh] = None,
        width: int = 1280,
        height: int = 720,
        title: str = "Ship Sensor Offsets",
    ) -> None:
        config = pyglet.gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
        super().__init__(width=width, height=height, caption=title, config=config, resizable=True)
        self.record = record
        self.mesh = mesh or self._load_mesh()
        self.camera = OrbitCamera()
        self._dragging = False
        self._last_mouse = (0, 0)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.05, 0.08, 0.12, 1.0)

    def _load_mesh(self) -> Optional[InventorMesh]:
        if not self.record.ship.model_path:
            return None
        return load_inventor(self.record.ship.model_path)

    def on_draw(self) -> None:  # pragma: no cover - requires OpenGL context
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / float(self.height or 1)
        gluPerspective(60.0, aspect, 0.1, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        self.camera.apply()

        self._draw_axes()
        if self.mesh is not None:
            self._draw_mesh(self.mesh)
        self._draw_sensors(self.record.sensors)

    def _draw_axes(self) -> None:
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(5.0, 0.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 5.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 5.0)
        glEnd()

    def _draw_mesh(self, mesh: InventorMesh) -> None:
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(0.4, 0.4, 0.45)
        glBegin(pyglet.gl.GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face:
                vx, vy, vz = mesh.vertices[idx]
                glVertex3f(vx, vy, vz)
        glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)

    def _draw_sensors(self, sensors: Iterable[SensorOffset]) -> None:
        glPointSize(8.0)
        glBegin(GL_POINTS)
        for sensor in sensors:
            glColor3f(1.0, 0.8, 0.1)
            glVertex3f(sensor.dx, sensor.dy, sensor.dz)
        glEnd()
        glBegin(GL_LINES)
        for sensor in sensors:
            glColor3f(1.0, 0.8, 0.1)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(sensor.dx, sensor.dy, sensor.dz)
        glEnd()

    def on_resize(self, width: int, height: int):  # pragma: no cover - requires OpenGL context
        super().on_resize(width, height)
        pyglet.gl.glViewport(0, 0, width, height)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            self.camera.azimuth += dx * 0.5
            self.camera.elevation = max(-89.0, min(89.0, self.camera.elevation + dy * 0.5))
        elif buttons & pyglet.window.mouse.RIGHT:
            self.camera.distance = max(1.0, self.camera.distance - dy * 0.05)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        self.camera.distance = max(1.0, self.camera.distance - scroll_y * 0.5)


def visualize(record: OffsetRecord) -> None:
    mesh: Optional[InventorMesh] = None
    if record.ship.model_path and Path(record.ship.model_path).exists():
        mesh = load_inventor(record.ship.model_path)
    viewer = OffsetViewer(record=record, mesh=mesh)
    pyglet.app.run()
