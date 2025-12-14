import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Check for sibling MTFLibrary and add to path if present
mtflib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../MTFLibrary/src")
)
if os.path.exists(mtflib_path):
    sys.path.insert(0, mtflib_path)
