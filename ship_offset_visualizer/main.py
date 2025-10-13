"""Command line interface for managing ship sensor offsets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .data import OffsetRecord, SensorOffset, Ship
from .io import OffsetRepository, record_from_csv
from .viewer import visualize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ship sensor offset management")
    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("add", help="Add a new sensor offset interactively")
    add_parser.add_argument("repo", type=Path, help="Repository path for saved records")
    add_parser.add_argument("record", type=str, help="Record name")
    add_parser.add_argument("sensor", type=str, help="Sensor name")
    add_parser.add_argument("dx", type=float, help="Offset along X")
    add_parser.add_argument("dy", type=float, help="Offset along Y")
    add_parser.add_argument("dz", type=float, help="Offset along Z")

    csv_parser = sub.add_parser("import", help="Import offsets from CSV")
    csv_parser.add_argument("repo", type=Path)
    csv_parser.add_argument("record", type=str)
    csv_parser.add_argument("ship", type=str)
    csv_parser.add_argument("csv", type=Path)
    csv_parser.add_argument("--model", type=Path, help="Path to ship Inventor model")
    csv_parser.add_argument("--description", type=str, default="")

    view_parser = sub.add_parser("view", help="Visualize a record")
    view_parser.add_argument("repo", type=Path)
    view_parser.add_argument("record", type=str)

    return parser


def cmd_add(repo_path: Path, record_name: str, sensor_name: str, dx: float, dy: float, dz: float) -> Path:
    repo = OffsetRepository(repo_path)
    record = repo.load(record_name) if record_name in repo.list_records() else OffsetRecord(ship=Ship(name=record_name))
    record.add_sensor(SensorOffset(sensor_name, dx, dy, dz))
    return repo.save(record, record_name)


def cmd_import(repo_path: Path, record_name: str, ship_name: str, csv_path: Path, model: Optional[Path], description: str) -> Path:
    ship = Ship(name=ship_name, model_path=model, description=description)
    record = record_from_csv(ship, csv_path)
    repo = OffsetRepository(repo_path)
    return repo.save(record, record_name)


def cmd_view(repo_path: Path, record_name: str) -> None:
    repo = OffsetRepository(repo_path)
    record = repo.load(record_name)
    visualize(record)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "add":
        path = cmd_add(args.repo, args.record, args.sensor, args.dx, args.dy, args.dz)
        print(f"Saved record to {path}")
    elif args.command == "import":
        path = cmd_import(args.repo, args.record, args.ship, args.csv, args.model, args.description)
        print(f"Imported record to {path}")
    elif args.command == "view":
        cmd_view(args.repo, args.record)
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
