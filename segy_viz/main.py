"""CLI entry point for the RadEx-style SEG-Y viewer."""

from __future__ import annotations

import argparse
import pathlib
import sys

from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow
from .display_modes import DisplayMode
from .colormap import ColormapType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RadEx-style SEG-Y viewer")
    parser.add_argument("path", type=pathlib.Path, nargs="?", help="Optional SEG-Y file to open")
    colormap_choices = [c.name.lower() for c in ColormapType]
    parser.add_argument(
        "--colormap",
        default="seismic",
        choices=colormap_choices,
        help="Default colormap",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Optional limit on traces to load",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the window title",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    window = MainWindow()
    if args.title:
        window.setWindowTitle(args.title)
    window.set_max_traces(args.max_traces)
    window.set_default_colormap(args.colormap)
    window.set_default_mode(DisplayMode.VARIABLE_DENSITY)
    window.show()

    if args.path:
        window.open_file(args.path)

    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
