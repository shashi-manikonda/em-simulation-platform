# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-09-27

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
- **CI Build Failure**: Resolved CI failures by removing the unsupported Python 3.13 from the test matrix and pinning the `mtflib` dependency to `v1.5.1`.
- **Documentation Build Script**: Corrected a path issue in `docs/build_docs.sh` that was causing CI failures.
- **Plotting TypeError**: Fixed a `TypeError` in `em_app/plotting.py` that occurred when generating heatmaps with complex number data.