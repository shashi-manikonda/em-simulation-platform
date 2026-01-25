# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - 2026-01-24

### Added
- **Benchmarks**: Added `stress_test_memory.py` to verify memory stability.

### Changed
- **Infrastructure**: Updated core dependency to `sandalwood>=0.1.2` (leveraging Memory Pooling).
- **Testing**: Implemented "Dual-Mode" demo verification (Quick/Full) with optimized regex patching.
- **CI/CD**: Added Pre-commit hooks for automated demo verification.

## [v0.2.8] - 2026-01-18

### Fixed
- **MPI Compatibility**: Resolved `mpi4py` segmentation fault during test collection by implementing a robust subprocess availability check.
- **Plotting Stability**: Fixed a crash in the COSY backend ("WRONG TYPE") during plotting by optimizing scalar multiplication of constant MTFs.
- **API Improvements**: Updated `VectorField` to support direct iteration and index access (`__getitem__`), fixing compatibility with demo scripts and plotting functions that previously accessed private attributes.
- **Test Suite**: Restored full pass rate for `tests/test_solvers.py`, `tests/test_demos.py`, and `tests/test_plotting.py`.

## [v0.2.7] - 2025-12-14

### Fixed
- **CI**: Added `contents: write` permission to GitHub Actions workflow to enable automated GitHub Release creation.

## [v0.2.6] - 2025-12-14

### Fixed
- **Release**: Bumped version to retry PyPI deployment after v0.2.5 failure.

## [v0.2.5] - 2025-12-14

### Changed
- **Demos**: Moved `05_ring_coil_demo.py` to `demos/em/` to ensure it is covered by the `run_all_demos.py` script.
- **Maintenance**: Applied `ruff` linting fixes (variable renaming, formatting) across the codebase.

## [v0.2.4] - 2025-12-14

### Fixed
- **Documentation**: Corrected the demo execution command in `README.md` to point to the new `scripts/` directory location.

## [v0.2.3] - 2025-12-14

### Added
- **RingCoil Support**: Fully integrated `RingCoil` source, including discretization logic and a new verification demo (`demos/05_ring_coil_demo.py`).

### Changed
- **Performance**: Optimized C++ backend integration for Biot-Savart calculations.
- **Dependencies**: Added `pandas-stubs` for type checking compliance.

### Fixed
- **Plotting**: Resolved 47 `ComplexWarning`s by strictly handling complex-to-real casting in plotting functions.
- **Type Safety**: Fixed `Vector` arithmetic to preserve subclass types (critical for `FieldVector`).
- **Cleanliness**: Resolved all linting (`ruff`) and type-checking (`mypy`) issues.

## [v0.2.2] - 2025-09-27

### Fixed
- **PyPI Publish Workflow**: Patched the `publish.yml` workflow to grant the correct permissions, resolving the "permission denied" error during PyPI publishing.

## [0.2.0] - 2025-09-27

### Added
- **Automated Dependency Updates**: Configured Dependabot to automatically manage `pip` and `GitHub Actions` dependencies.
- **Automated PyPI Publishing**: Added a new GitHub Actions workflow (`publish.yml`) to automatically build and publish the package to PyPI when a new version tag is pushed.
- **Expanded Test Suite**: Added unit tests for the `em_app/plotting.py` module to improve test coverage.
- **Release Documentation**: Added a `RELEASE.md` file with detailed instructions for the release process.

### Changed
- **CI/CD Pipeline**: Enhanced the `ci.yml` workflow to include steps for building the package and documentation, ensuring they are validated on every change.
- **Release Process**: The release process is now partially automated. Pushing a version tag triggers the PyPI publish workflow.

### Fixed
- **CI Build Failure**: Resolved CI failures by removing the unsupported Python 3.13 from the test matrix and pinning the `sandalwood` dependency to `v1.5.1`.
- **Documentation Build Script**: Corrected a path issue in `docs/build_docs.sh` that was causing CI failures.
- **Plotting TypeError**: Fixed a `TypeError` in `em_app/plotting.py` that occurred when generating heatmaps with complex number data.