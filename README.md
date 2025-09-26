# em-simulation-platform

EM simulation tools for electromagnetic field analysis, visualization, and benchmarking.

## Features
- Modular solvers for EM field calculations
- Source modeling (dipoles, wires, loops, solenoids)
- Advanced plotting and visualization
- Demo scripts for validation and exploration
- Benchmarking utilities
- Extensible architecture for research and teaching

## Installation

```bash
# Clone the repository
git clone https://github.com/shashi-manikonda/em-simulation-platform.git
cd em-simulation-platform

# (Recommended) Create a virtual environment
python3 -m venv .mtfvenv
source .mtfvenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run all demos
```bash
python run_all_demos.py
```

### Run a specific demo
```bash
python demos/em/01_validation_demo.py
```

### Run tests
```bash
pytest
```

## Example: Calculate and Plot Magnetic Field of a Dipole
```python
from src.em_app.sources import Dipole
from src.em_app.plotting import plot_field

# initialize mtf
mtf.initialize_mtf(max_order=6, max_dimension=4)

# Create a dipole source
source = Dipole(position=[0,0,0], moment=[0,0,1])

# Calculate field at grid points
field = source.calculate_field(grid_points)

# Plot the field
plot_field(field)
```

## Project Structure
- `src/em_app/` - Core library modules
- `demos/em/` - Demo scripts
- `benchmarks/` - Performance and accuracy benchmarks
- `tests/` - Unit tests

## License
MIT
