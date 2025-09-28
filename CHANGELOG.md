# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-09-27

### Added
- **Pre-Commit Hooks**: Set up `pre-commit` with `ruff` to automatically enforce code quality and formatting.
- **Automated Dependency Updates**: Configured Dependabot to automatically manage `pip` and `GitHub Actions` dependencies.
- **Automated PyPI Publishing**: Added a new GitHub Actions workflow (`publish.yml`) to automatically build, publish, and create releases on PyPI and GitHub.
- **Expanded Test Suite**: Added unit tests for the `em_app/plotting.py` module to improve test coverage.
- **Sphinx Example Gallery**: Created a gallery of examples in the documentation, automatically generated from the demo scripts.
- **Release Documentation**: Added a `RELEASE.md` file with detailed instructions for the release process.

### Changed
- **CI/CD Pipeline**: Enhanced the `ci.yml` workflow to include steps for building the package and documentation.
- **Release Process**: The release process is now fully automated. Pushing a version tag triggers the PyPI publish and GitHub Release creation workflow.
- **README**: Improved the `README.md` with a more prominent "Quick Start" section.
- **Demo Scripts**: Converted all demo Jupyter Notebooks to standard Python scripts to simplify execution and enable `sphinx-gallery` integration.

### Fixed
- **CI Build Failure**: Resolved CI failures by removing the unsupported Python 3.13 from the test matrix and pinning the `mtflib` dependency to `v1.5.1`.
- **Documentation Build**: Fixed multiple issues with the Sphinx documentation build, including a path issue in `docs/build_docs.sh` and incorrect gallery configurations in `docs/conf.py` and `docs/gallery.rst`.
- **Plotting TypeError**: Fixed a `TypeError` in `em_app/plotting.py` that occurred when generating heatmaps with complex number data.
- **Test Performance**: Optimized plotting tests to run quickly and reliably, resolving CI timeouts.
- **Version Control**: Removed generated artifacts (`runoutput/`, `docs/auto_examples/`) from version control and added them to `.gitignore`.
