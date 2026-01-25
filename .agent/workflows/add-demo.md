---
description: Onboarding a new Demo script
---

Procedure for integrating new research code ensuring it doesn't break CI.

1. Place the new script in `demos/[category]/[name].py`.
2. Create or update an entry in `tests/test_demos.py` to include the new demo in the test discovery.
3. Verify the "Quick Mode" patching works for this demo (low order, minimal segments).
4. // turbo
   Run integration check:
   ```bash
   pytest -v -m demo tests/test_demos.py
   ```
