---
description: Total Rebuild of HPC Kernels and Environment
---

Use this when moving to a new machine or modifying the `sandalwood` Fortran core.

1. Verify `sandalwood` repository exists at `../sandalwood`.
2. // turbo
   Run the Linux build script (or Windows equivalent):
   ```bash
   bash scripts/linux/setup_dev_env.sh
   ```
3. Verify binary linking (Linux):
   ```bash
   ldd ../sandalwood/src/sandalwood/backends/cosy/libcosy.so
   ```
4. Confirm backend availability in the app:
   ```bash
   pytest tests/test_solvers.py
   ```
