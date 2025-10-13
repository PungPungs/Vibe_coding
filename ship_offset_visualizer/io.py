"""Utilities for reading and writing offset records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from .data import OffsetRecord, SensorOffset, Ship


class OffsetRepository:
    """High level access to stored offset records."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: OffsetRecord, name: str) -> Path:
        path = self.root / f"{name}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(record.to_dict(), fp, indent=2)
        return path

    def load(self, name: str) -> OffsetRecord:
        path = self.root / f"{name}.json"
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return OffsetRecord.from_dict(data)

    def list_records(self) -> List[str]:
        return [p.stem for p in self.root.glob("*.json")]


def record_from_csv(ship: Ship, csv_path: Path) -> OffsetRecord:
    """Load sensor offsets from a CSV file."""

    sensors: List[SensorOffset] = []
    with Path(csv_path).open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            sensors.append(
                SensorOffset(
                    name=row.get("name", ""),
                    dx=float(row.get("dx", 0.0)),
                    dy=float(row.get("dy", 0.0)),
                    dz=float(row.get("dz", 0.0)),
                    metadata={k: v for k, v in row.items() if k not in {"name", "dx", "dy", "dz"}},
                )
            )
    return OffsetRecord(ship=ship, sensors=sensors)


def record_to_csv(record: OffsetRecord, path: Path) -> Path:
    """Export a record to CSV."""

    fieldnames = {"name", "dx", "dy", "dz"}
    for sensor in record.sensors:
        fieldnames.update(sensor.metadata.keys())
    ordered_fields = ["name", "dx", "dy", "dz"] + sorted(fieldnames - {"name", "dx", "dy", "dz"})

    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=ordered_fields)
        writer.writeheader()
        for sensor in record.sensors:
            row = {"name": sensor.name, "dx": sensor.dx, "dy": sensor.dy, "dz": sensor.dz}
            row.update(sensor.metadata)
            writer.writerow(row)
    return Path(path)
