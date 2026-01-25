# Developer Guide: EM Simulation Platform

Welcome! This document provides information for developers who wish to contribute to the **EM Simulation Platform**.

## 🛠️ Environment Setup

We use `uv` for lightning-fast dependency management and `pre-commit` for quality assurance.

### 1. Unified Setup (Recommended)
Run the automated setup script for your platform:
- **Linux**: `./scripts/linux/setup_dev_env.sh`
- **Windows**: `scripts/windows/setup_env.bat`

These scripts will:
* Create a virtual environment (`.venv`).
* Install `uv`.
* Install `sandalwood` and `em-app` in editable mode.
* **Automatically install the `pre-commit` hooks.**

### 2. Manual Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -e .[dev,benchmark]
pre-commit install
```

## 🧪 Testing Workflow

The platform uses a **Dual-Mode** testing strategy to balance development speed with physical fidelity.

### 1. Standard Tests (Fast)
```bash
pytest
```
*   **Target**: Logic, data structures, and small-scale physics.
*   **Excluded**: All demos and slow simulations (marked with `@pytest.mark.demo`).

### 2. Demo Verification (v0.3.0 Feature)
We verify that all demos in the `demos/` directory run without crashes.

*   **Quick Mode (Automatic)**:
    When running `pytest -m demo tests/test_demos.py`, the system automatically patches demos to use lower orders and fewer segments (~30s total).
*   **Full Mode (Exhaustive)**:
    To run the demos with their original, high-fidelity physics parameters:
    ```bash
    export EM_APP_TEST_FULL_DEMOS=1
    pytest -v -m demo tests/test_demos.py
    ```

## 📈 Benchmarking

To ensure the platform remains stable under load, we use dedicated benchmarks.

### Memory Stability
```bash
python benchmarks/stress_test_memory.py
```
This script monitors RSS memory usage during millions of COSY index allocations/deallocations.
*   **Pass Condition**: Memory growth should be zero (flatline) after the initial pool warming (~1024 indices).

## 🧹 Code Quality

We enforce strict linting and type checking:
*   **Linting**: `ruff check .`
*   **Formatting**: `ruff format .`
*   **Type Checking**: `mypy src` (Handles `sandalwood` and `mpi4py` overlays).

## 🏗️ Architecture Note

When adding new solvers, always use the **Structure of Arrays (SoA)** pattern. Pass raw arrays for `Bx`, `By`, and `Bz` to the `VectorField` constructor to leverage high-performance batch operations in the `sandalwood` backend.
