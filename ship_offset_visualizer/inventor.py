"""Minimal Open Inventor ``.iv`` parser for geometry extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


_COORD_PATTERN = re.compile(r"Coordinate3\s*\{[^}]*point\s+\[([^\]]+)\]", re.MULTILINE | re.DOTALL)
_INDEX_PATTERN = re.compile(r"IndexedFaceSet\s*\{[^}]*coordIndex\s+\[([^\]]+)\]", re.MULTILINE | re.DOTALL)


@dataclass
class InventorMesh:
    vertices: np.ndarray
    faces: np.ndarray


def _parse_floats(text: str) -> List[Tuple[float, float, float]]:
    triples: List[Tuple[float, float, float]] = []
    for line in text.split(","):
        parts = line.strip().split()
        if len(parts) == 3:
            triples.append(tuple(float(p) for p in parts))
    return triples


def _parse_indices(text: str) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    current: List[int] = []
    for token in text.replace("\n", " ").split(","):
        token = token.strip()
        if not token:
            continue
        if token == "-1":
            if len(current) >= 3:
                # triangulate fan
                for i in range(1, len(current) - 1):
                    faces.append((current[0], current[i], current[i + 1]))
            current = []
            continue
        current.append(int(token))
    return faces


def load_inventor(path: Path) -> InventorMesh:
    """Load a minimal subset of Inventor geometry."""

    data = Path(path).read_text(encoding="utf-8")
    coord_match = _COORD_PATTERN.search(data)
    index_match = _INDEX_PATTERN.search(data)
    if not coord_match or not index_match:
        raise ValueError("The .iv file must contain Coordinate3 and IndexedFaceSet nodes")

    vertices = np.array(_parse_floats(coord_match.group(1)), dtype=float)
    faces = np.array(_parse_indices(index_match.group(1)), dtype=int)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("Failed to parse geometry from Inventor file")
    return InventorMesh(vertices=vertices, faces=faces)
