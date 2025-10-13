"""Core data structures for ship and sensor offset handling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class SensorOffset:
    """Represents a single sensor offset measurement."""

    name: str
    dx: float
    dy: float
    dz: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_vector(self) -> np.ndarray:
        """Return the offset as a 3-element numpy vector."""

        return np.array([self.dx, self.dy, self.dz], dtype=float)


@dataclass
class Ship:
    """Definition of a ship reference frame and geometry."""

    name: str
    model_path: Optional[Path] = None
    description: str = ""


@dataclass
class OffsetRecord:
    """Collection of sensor offsets for a specific ship."""

    ship: Ship
    sensors: List[SensorOffset] = field(default_factory=list)

    def add_sensor(self, offset: SensorOffset) -> None:
        self.sensors.append(offset)

    def extend(self, offsets: Iterable[SensorOffset]) -> None:
        self.sensors.extend(offsets)

    def as_matrix(self) -> np.ndarray:
        """Return all offsets stacked as an ``(n, 3)`` matrix."""

        if not self.sensors:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([sensor.as_vector() for sensor in self.sensors])

    def to_dict(self) -> Dict:
        return {
            "ship": {
                "name": self.ship.name,
                "model_path": str(self.ship.model_path) if self.ship.model_path else None,
                "description": self.ship.description,
            },
            "sensors": [
                {
                    "name": sensor.name,
                    "dx": sensor.dx,
                    "dy": sensor.dy,
                    "dz": sensor.dz,
                    "metadata": sensor.metadata,
                }
                for sensor in self.sensors
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "OffsetRecord":
        ship_data = payload["ship"]
        ship = Ship(
            name=ship_data.get("name", ""),
            model_path=Path(ship_data["model_path"]) if ship_data.get("model_path") else None,
            description=ship_data.get("description", ""),
        )
        sensors = [
            SensorOffset(
                name=item.get("name", ""),
                dx=float(item.get("dx", 0.0)),
                dy=float(item.get("dy", 0.0)),
                dz=float(item.get("dz", 0.0)),
                metadata=item.get("metadata", {}) or {},
            )
            for item in payload.get("sensors", [])
        ]
        return cls(ship=ship, sensors=sensors)
