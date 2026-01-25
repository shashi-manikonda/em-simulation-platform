# Agent Rules: EM Simulation Platform

These rules ensure the high-performance and architectural standards of the **EM Simulation Platform** (v0.3.0+) are maintained.

## 🏗️ Architecture & Performance

*   **SoA Enforcement**: When implementing new EM solvers or vector operations, always use the **Structure of Arrays (SoA)** pattern (separate contiguous arrays for x, y, z) rather than lists of objects. This ensures compatibility with vectorized Numba and Fortran kernels in the `sandalwood` backend.
*   **Vectorized Dispatch**: Ensure that `VectorField` operations prioritize the internal `_get_components()` helper to avoid the overhead of iterating over `FieldVector` objects in Python.
*   **MPI Result Integrity**: Parallel calculations (e.g., in `mpi_biot_savart`) must always broadcast results from the root rank to all other ranks using `comm.bcast`. Failure to do so leads to `NoneType` crashes on non-root processes during downstream operations.
*   **Memory Benchmarking**: Any change to the core calculation loop or long-running simulations must be verified with `benchmarks/stress_test_memory.py` to ensure no memory growth/leaks.

## 🌉 Workspace & Environment

*   **Workspace Sensitivity**: When working on features requiring `sandalwood` changes, ensure a local editable installation is used from the adjacent workspace directory (`../sandalwood`).
*   **Linux Setup Authority**: The single source of truth for the Linux development environment is `/home/mls/work/em-simulation-platform/setup_dev_env.sh`. This script MUST handle:
    *   Building the COSY backend.
    *   Activating the virtual environment.
    *   Installing `em-app` and `sandalwood`.
    *   Any modifications to the build/install process MUST be made to this file first.
    *   A copy MUST be maintained at `scripts/linux/setup_dev_env.sh`.

## 🧪 Testing & Documentation

*   **Demo Isolation**: strictly apply `@pytest.mark.demo` to any test that runs a full demo file. Ensure that the standard `pytest` run (standard tier) remains fast (<5s).
*   **Dual-Mode Verification**: When modifying `tests/test_demos.py`, verify that the "Quick Mode" regex patching logic remains robust and correctly accelerates the demos for CI cycles.
*   **Documentation Traceability**: Every new coil type or physical solver MUST be documented in `docs/theory.rst` with its corresponding integral form or multipole approximation logic.
*   **Vector Type Consistency**: Prefer `FieldVector` (from `em_app.vector_fields`) over base `Vector` or raw NumPy arrays for physical fields to ensure metadata (like units or MTF status) is preserved during transformations.

## 🧹 Code Quality

*   **Dependency Hygiene**: Always use `uv` for package management operations.
*   **Linting & Typing**: Before pushing, always run `ruff check .` and `mypy src` to ensure code quality standards are met. This is enforced by CI.
*   **Automation**: Prioritize using the established setup scripts in `scripts/` (e.g., `setup_dev_env.sh`) for environment initialization to ensure `pre-commit` hooks and MPI paths are correctly configured.
