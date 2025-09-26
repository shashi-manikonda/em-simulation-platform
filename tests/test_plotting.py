import numpy as np
import pytest
import matplotlib.pyplot as plt
from em_app.sources import RingCoil
from em_app.plotting import plot_1d_field, plot_2d_field, plot_field_vectors_3d

@pytest.fixture(scope="module")
def ring_coil():
    """Create a simple RingCoil for plotting tests."""
    return RingCoil(current=1.0, radius=1.0, num_segments=20)


def test_plot_1d_field_smoke(ring_coil):
    """
    Smoke test for plot_1d_field to ensure it runs without errors.
    """
    try:
        fig, ax = plt.subplots()
        plot_1d_field(ring_coil, field_component='z', ax=ax)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_1d_field raised an exception: {e}")


def test_plot_2d_field_smoke(ring_coil):
    """
    Smoke test for plot_2d_field to ensure it runs without errors.
    """
    try:
        fig, ax = plt.subplots()
        plot_2d_field(ring_coil, field_component='norm', plot_type='heatmap', ax=ax)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_2d_field raised an exception: {e}")


def test_plot_field_vectors_3d_smoke(ring_coil):
    """
    Smoke test for plot_field_vectors_3d to ensure it runs without errors.
    """
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_field_vectors_3d(ring_coil, ax=ax)
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_field_vectors_3d raised an exception: {e}")