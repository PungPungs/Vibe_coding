# Ship Offset Visualizer

Python toolkit for capturing ship sensor offset measurements, storing them in a
repository, and visualising the offsets together with the ship's 3D geometry.
The viewer is built on top of `pyglet`/OpenGL to keep rendering fast even with
larger models.

## Features

- Record individual sensor offsets or bulk import from CSV.
- Persist measurements in JSON within a project repository.
- Load Open Inventor (`.iv`) surface models and render them alongside the
  measured offsets.
- GPU accelerated visualisation using OpenGL with anti-aliasing and orbit
  controls.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The package intentionally keeps dependencies minimal:

- `numpy`
- `pyglet`

## Usage

### Creating or updating records

Import offsets from a CSV file (columns: `name, dx, dy, dz` plus optional
metadata fields):

```bash
python -m ship_offset_visualizer.main import data_repo survey_a "Survey Vessel" offsets.csv \
  --model ship.iv --description "2024 multibeam survey"
```

Add or update a single sensor offset:

```bash
python -m ship_offset_visualizer.main add data_repo survey_a usbL 0.42 -0.15 1.78
```

### Visualisation

Launch the viewer for a stored record:

```bash
python -m ship_offset_visualizer.main view data_repo survey_a
```

Use the left mouse button to orbit, the right button (or scroll wheel) to
change the distance.

## Notes on Inventor support

The included Inventor parser handles ASCII `.iv` files that provide
`Coordinate3` and `IndexedFaceSet` nodes. More advanced material or hierarchy
features can be added later by extending `inventor.py`.

## Development

Run the module directly in editable mode during development:

```bash
python -m ship_offset_visualizer.main --help
```
