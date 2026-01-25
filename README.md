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
uv pip install -e .[dev,benchmark]
```

4. **Install Git Hooks:**
```bash
pre-commit install
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

1. **Standard (Fast):**
```bash
pytest
```
*Runs unit tests only; excludes slow demos.*

2. **Commit Check (Quick Demos):**
```bash
pre-commit run --all-files
```
*Runs the "Quick" version of demo verification.*

3. **Full Verification (Slow):**
```bash
export EM_APP_TEST_FULL_DEMOS=1
pytest tests/test_demos.py
```
*Runs full physics simulations (unmodified demos).*

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
from sandalwood import mtf

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
## Architecture & Optimizations

This platform employs several advanced design patterns and algorithms to ensure high performance and flexibility:

### Data Structures
*   **Structure of Arrays (SoA)**: The `VectorField` class detects input formats and switches to SoA storage (`_storage_mode = "soa"`) when initialized with component arrays (vx, vy, vz). This improves memory locality and SIMD vectorization potential compared to Array of Structures (AoS).
*   **Hybrid Storage**: Seamlessly handles both numerical data (NumPy arrays) and symbolic objects (`sandalwood` MTFs) within the same API.

### Design Patterns
*   **Factory Pattern**: `Vector.from_array_of_vectors` provides optimizing factory methods for bulk object creation.
*   **Strategy/Adapter Pattern**: The `solvers` module uses a backend selection strategy (`Backend.PYTHON`, `Backend.COSY`, `Backend.MPI`) to dispatch computation to the most appropriate engine (local CPU, optimized Fortran, or distributed MPI).

## Windows Installation & Development Guide

This project (and its dependency `sandalwood`) requires specific setup on Windows to support the high-performance Fortran COSY backend.

### 1. Prerequisites
*   **Python 3.9+**
*   **Git**
*   **Visual Studio Build Tools 2022**: Ensure "Desktop development with C++" is selected during installation. This provides `link.exe` and `nmake`.
*   **Intel oneAPI HPC Kit**: Required for the `ifx` Fortran compiler.
*   **Intel oneAPI Base Kit**: Required for Intel MPI libraries.

### 2. Workspace Setup (Recommended)
We recommend setting up a common workspace for both `sandalwood` and `em-simulation-platform` to share a virtual environment.

```powershell
# Directory Structure
# C:\Users\YourName\Work\
#   ├── sandalwood/
#   ├── em-simulation-platform/
#   └── .venv/  (or inside one repo)
```

### 3. Step-by-Step Installation (Automated)

We provide a **unified setup script** that handles the complex environment configuration (detecting Visual Studio, setting up Intel compilers, installing libraries, and building extensions).

**Method A: Automated Setup (Recommended)**
```powershell
cd C:\Users\YourName\Work\DAProjects
.\em-simulation-platform\scripts\windows\setup_env.bat
```
*This script will:*
1.  Initialize Intel oneAPI environment (`setvars.bat`).
2.  Set up Visual Studio integration.
3.  Create/Update the `.venv`.
4.  Build `sandalwood` (compiling Fortran with `ifx`).
5.  Install `em-simulation-platform`.

**Method B: Manual Installation**
If you prefer manual control:

**Step 1: Clone Repositories**
```powershell
cd C:\Users\YourName\Work
git clone https://github.com/shashi-manikonda/sandalwood.git
git clone https://github.com/shashi-manikonda/em-simulation-platform.git
```

**Step 2: Setup Virtual Environment**
It is easiest to use a single virtual environment for both projects.
```powershell
cd sandalwood
uv venv .venv
# Activate
.venv\Scripts\activate
```

**Step 3: Build and Install Sandalwood (The Compiler Step)**
This step compiles the Fortran backend (`libcosy.dll`).
```powershell
# Ensure you are in the sandalwood directory
# Note: You MUST have 'ifx' and 'link.exe' in your PATH (run 'setvars.bat' first)
uv pip install -e .[dev]
```

**Step 4: Install EM Platform**
```powershell
cd ..\em-simulation-platform
# Install into the SAME virtual environment
uv pip install -e .[dev,benchmark]
```

**Step A: Clone Repositories**
```powershell
cd C:\Users\YourName\Work
git clone https://github.com/shashi-manikonda/sandalwood.git
git clone https://github.com/shashi-manikonda/em-simulation-platform.git
```

**Step B: setup Virtual Environment**
It is easiest to use a single virtual environment for both projects.
```powershell
cd sandalwood
uv venv .venv
# Activate
.venv\Scripts\activate
```

**Step C: Build and Install Sandalwood (The Compiler Step)**
This step compiles the Fortran backend (`libcosy.dll`).
```powershell
# Ensure you are in the sandalwood directory
uv pip install -e .[dev]
```
*   **Note**: The build script (`setup.py`) will automatically detect your Visual Studio and Intel compilers.
*   **Troubleshooting**: If you see linker errors, ensure you have the Intel Base Kit installed and `libiomp5md.lib` is reachable.

**Step D: Install EM Platform**
```powershell
cd ..\em-simulation-platform
# Install into the SAME virtual environment
uv pip install -e .[dev,benchmark]
```

### 4. Working with the COSY Backend
The COSY backend is a compiled Fortran extension.

*   **Modifying Fortran Code**: If you modify `src/sandalwood/backends/cosy/wrapper.f` in the `sandalwood` repo, **changes do not take effect automatically**. You must re-compile:
    ```powershell
    cd ..\sandalwood
    uv pip install -e .
    ```
    This triggers the custom build command to regenerate the DLL.

*   **Memory Configuration (Windows vs Linux)**:
    Windows has a 2GB limit for static object file sections. `sandalwood` handles this automatically via `src/sandalwood/backends/cosy/cosy_config.env`:
    *   **Linux**: `COSY_LMEM` defaults to 1GB (allows large simulations).
    *   **Windows**: `COSY_LMEM_WIN32` overrides this to ~150MB to fit within OS limits.
    *   If you need more memory on Windows, consider using the Linux Subsystem for Windows (WSL2).

### 5. Running Tests
```powershell
cd ..\em-simulation-platform
pytest
```

## Advanced Configuration: MPI and Linux
(See previous documentation for Linux-specific bash instructions)
[...MPI instructions remain similar...]

