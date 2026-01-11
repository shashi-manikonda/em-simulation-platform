import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Check for sibling sandalwood and add to path if present
sandalwood_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../sandalwood/src")
)
if os.path.exists(sandalwood_path):
    sys.path.insert(0, sandalwood_path)

from sandalwood import mtf  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def initialize_sandalwood():
    """Initializes Sandalwood MTF once for the entire test session."""
    # Using max order and dimension used in tests
    mtf.initialize_mtf(max_order=5, max_dimension=4, implementation="cosy")
    # Default tolerance
    mtf.set_etol(1e-20)
    yield


@pytest.fixture(scope="function", autouse=True)
def reset_sandalwood():
    """Resets Sandalwood MTF memory between tests."""
    yield
    mtf.reset_mtf()
