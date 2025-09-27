# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-09-27

### Fixed
- The CI build was failing due to an unsupported Python version (3.13) in the test matrix. This version has been removed from the CI configuration.
- The `mtflib` dependency is now pinned to version `1.5.1` in `pyproject.toml` to ensure reproducible builds from PyPI.