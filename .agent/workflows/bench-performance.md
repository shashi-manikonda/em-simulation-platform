---
description: Performance Benchmarking of EM Solvers
---

Run this workflow to verify that code changes haven't regressed calculation speed or memory usage.

1. Ensure the COSY backend is initialized:
   ```bash
   pytest tests/test_solvers.py
   ```
2. // turbo
   Run the Memory Stress Test (Checking for leaks):
   ```bash
   python benchmarks/stress_test_memory.py
   ```
3. // turbo
   Run the Speed Benchmark:
   ```bash
   python benchmarks/biot_savart_benchmark.py
   ```
4. Compare performance against the baseline in `README.md`.
