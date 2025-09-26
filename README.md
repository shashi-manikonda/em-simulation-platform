# EM Simulation Platform

EM Simulation Platform is a Python package for calculating and visualizing electromagnetic fields from various current sources. It is designed to be flexible and extensible, with a focus on using Multivariate Taylor Functions (MTFs) from the `mtflib` library for advanced mathematical analysis.

## Key Features

*   **Biot-Savart Solvers:** Includes multiple backends for Biot-Savart calculations, including pure Python, C++, and a parallelized MPI version.
*   **Coil Geometries:** Provides classes for common coil geometries, such as circular rings, rectangles, and straight wires.
*   **Advanced Analysis:** Leverages `mtflib` to perform advanced operations like calculating the curl, divergence, and gradient of vector fields.
*   **Comprehensive Plotting:** Offers a range of plotting functions to visualize fields in 1D, 2D, and 3D.
*   **Extensible Design:** The object-oriented design makes it easy to add new coil types or field solvers.

## Documentation

For a complete guide to the package, including installation instructions, tutorials, and a detailed API reference, please see our full documentation.

**Note:** The documentation will be hosted at a future URL. For now, you can build it locally by navigating to the `docs` directory and running `make html`.

## Getting Started

To install the package and its dependencies, run the following command from the root of the repository:

```bash
pip install -e .
```

To include the development dependencies for running tests and building documentation, use:
```bash
pip install -e .[dev]
```

## Running Tests

To run the test suite, use `pytest`:
```bash
pytest
```