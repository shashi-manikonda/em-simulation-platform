# em-simulation-platform

[![Python CI](https://github.com/shashi-manikonda/em-simulation-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/shashi-manikonda/em-simulation-platform/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/em-app.svg)](https://badge.fury.io/py/em-app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/em-app/badge/?version=latest)](https://em-app.readthedocs.io/en/latest/?badge=latest)

EM simulation tools for electromagnetic field analysis, visualization, and benchmarking.

## Features
- Modular solvers for EM field calculations (Python, C, and Optimized C++ backends)
- Source modeling (dipoles, wires, loops/RingCoil, solenoids)
- Advanced plotting and visualization
- Demo scripts for validation and exploration
- Benchmarking utilities
- Extensible architecture for research and teaching

## Installation

This project uses ``pyproject.toml`` to manage dependencies. For development, it is recommended to install the package in "editable" mode along with the development extras.

```bash
# Clone the repository
git clone https://github.com/shashi-manikonda/em-simulation-platform.git
cd em-simulation-platform

# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with all development dependencies
pip install -e .[dev,benchmark]
```

The ``[dev]`` extra includes dependencies for running tests and building the documentation. The ``[benchmark]`` extra includes dependencies for running the benchmark scripts.

## Usage

### Run all demos
```bash
python scripts/run_all_demos.py
```

### Run a specific demo
To run a specific demo, you can execute the script directly. For notebooks, you can use a tool like `jupytext` to run it as a script:
```bash
jupytext --execute demos/em/01_validation_demo.ipynb
```

### Run tests
```bash
pytest
```

## Building the Documentation

This project uses Sphinx to generate API documentation from the source code. The necessary dependencies are included in the `[dev]` extra.

### Build Script

A helper script is provided to simplify the build process. To build the documentation, run the following command from the project root:

```bash
./docs/build_docs.sh
```

The script will clean the previous build and generate the HTML documentation in the `docs/_build/html` directory.

### Viewing the Documentation

To view the documentation, open the `docs/_build/html/index.html` file in your web browser.

## Example: Calculate and Plot the Magnetic Field of a Ring Coil

This example demonstrates how to define a current source, calculate its magnetic field on a grid, and visualize the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from em_app.sources import RingCoil
from em_app.solvers import calculate_b_field, Backend
from mtflib import mtf

# Initialize the MTF library (Optional - defaults to Order 4, Dim 3 if omitted)
# mtf.initialize_mtf(max_order=1, max_dimension=4)

# --- 1. Setup the Coil Geometry ---
coil = RingCoil(
    current=1.0,
    radius=0.5,
    num_segments=20,
    center_point=np.array([0, 0, 0]),
    axis_direction=np.array([0, 0, 1]),
)

# --- 2. Define the Field Points for Calculation ---
grid_size = 1.0
num_points = 15
x_points = np.linspace(-grid_size, grid_size, num_points)
z_points = np.linspace(-grid_size, grid_size, num_points)
X, Z = np.meshgrid(x_points, z_points)
field_points = np.vstack([X.ravel(), np.zeros_like(X.ravel()), Z.ravel()]).T

# --- 3. Calculate the Magnetic Field ---
# --- 3. Calculate the Magnetic Field ---
# You can specify the backend explicitly using the Backend Enum
b_field = calculate_b_field(coil, field_points, backend=Backend.PYTHON)
b_vectors = np.array([b.to_numpy_array() for b in b_field._vectors_mtf])

# --- 4. Plot the Results ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the coil geometry
coil.plot(ax, color="b", wire_thickness=0.02)

# Plot the magnetic field vectors
ax.quiver(
    field_points[:, 0],
    field_points[:, 1],
    field_points[:, 2],
    b_vectors[:, 0],
    b_vectors[:, 1],
    b_vectors[:, 2],
    length=0.2,
    normalize=True,
    color="gray",
)

# --- 5. Customize and Show the Plot ---
ax.set_title("Magnetic Field of a Ring Coil")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.view_init(elev=20.0, azim=-60)
plt.show()
```

## Live Demos

For more detailed examples, see the demo scripts in the `demos/em` directory. These scripts cover topics such as solver validation, dipole approximation, and advanced plotting.

You can run all demos at once using the following command:
```bash
python scripts/run_all_demos.py
```
This will generate output files and plots in the `runoutput` directory.

## Project Structure
- `src/em_app/` - Core library modules
- `demos/em/` - Demo scripts
- `benchmarks/` - Performance and accuracy benchmarks
- `tests/` - Unit tests

## License
MIT