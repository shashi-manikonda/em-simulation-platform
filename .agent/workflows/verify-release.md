---
description: Full QA and Release Verification for em-app
---

Standard procedure to verify v0.3.0+ releases.

1. Run standard unit tests: 
   ```bash
   pytest -v
   ```
2. Run memory stability benchmark: 
   ```bash
   python benchmarks/stress_test_memory.py
   ```
3. // turbo
   Verify demos in Quick Mode: 
   ```bash
   pytest -v -m demo tests/test_demos.py
   ```
4. Check linting and types:
   ```bash
   ruff check . && mypy src
   ```
5. Update `CHANGELOG.md` with the latest commit summaries and bump version in `pyproject.toml` if necessary.
