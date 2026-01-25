# Agent Rules: EM Simulation Platform

These rules ensure the high-performance and architectural standards of the **EM Simulation Platform** (v0.3.0+) are maintained.

## 🏗️ Architecture & Performance

*   **SoA Enforcement**: When implementing new EM solvers or vector operations, always use the **Structure of Arrays (SoA)** pattern (separate contiguous arrays for x, y, z) rather than lists of objects. This ensures compatibility with vectorized Numba and Fortran kernels in the `sandalwood` backend.
*   **Vectorized Dispatch**: Ensure that `VectorField` operations prioritize the internal `_get_components()` helper to avoid the overhead of iterating over `FieldVector` objects in Python.
*   **Memory Benchmarking**: Any change to the core calculation loop or long-running simulations must be verified with `benchmarks/stress_test_memory.py` to ensure no memory growth/leaks.

## 🧪 Testing Standards

*   **Demo Isolation**: strictly apply `@pytest.mark.demo` to any test that runs a full demo file. Ensure that the standard `pytest` run (standard tier) remains fast (<5s).
*   **Dual-Mode Verification**: When modifying `tests/test_demos.py`, verify that the "Quick Mode" regex patching logic remains robust and correctly accelerates the demos for CI cycles.

## 🧹 Code Quality

*   **Dependency Hygiene**: Always use `uv` for package management operations.
*   **Automation**: Prioritize using the established setup scripts in `scripts/` (e.g., `setup_dev_env.sh`) for environment initialization to ensure `pre-commit` hooks and MPI paths are correctly configured.
