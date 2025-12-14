import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mtflib import mtf

matplotlib.use("Agg")  # Use a non-interactive backend for testing

from em_app.plotting import (
    plot_1d_field,
    plot_2d_field,
    plot_field_vectors_3d,
)
from em_app.sources import RingCoil

# Global settings for tests
MAX_ORDER = 5
MAX_DIMENSION = 4
ETOL = 1e-20


@pytest.fixture(scope="function", autouse=True)
def setup_function():
    mtf.initialize_mtf(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION)
    mtf.set_etol(ETOL)
    global_dim = mtf.get_max_dimension()
    exponent_zero = tuple([0] * global_dim)
    yield global_dim, exponent_zero
    mtf._INITIALIZED = False
    plt.close("all")  # Ensure figures are closed after each test


# Fixture for a simple coil to be used in tests
@pytest.fixture
def coil():
    """A very simple coil fixture to ensure fast tests."""
    return RingCoil(
        center_point=np.array([0, 0, 0]),
        axis_direction=np.array([0, 0, 1]),
        radius=1.0,
        current=1.0,
        num_segments=4,  # Drastically reduced for speed
    )


# --- Tests for plot_1d_field ---
@pytest.mark.parametrize("field_component", ["x", "y", "z", "norm"])
@pytest.mark.parametrize("plot_type", ["line", "scatter"])
@pytest.mark.filterwarnings("ignore:Data has no positive values")
def test_plot_1d_field_runs_successfully(coil, field_component, plot_type):
    """Test that plot_1d_field runs without errors for various configurations."""
    try:
        plot_1d_field(
            coil,
            field_component=field_component,
            plot_type=plot_type,
            log_scale=True,
            num_points=2,  # Drastically reduced for speed
        )
        plot_1d_field(
            coil,
            field_component=field_component,
            start_point=np.array([-1, -1, -1]),
            end_point=np.array([1, 1, 1]),
            plot_type=plot_type,
            num_points=2,  # Drastically reduced for speed
        )
    except Exception as e:
        pytest.fail(f"plot_1d_field raised an unexpected exception: {e}")


def test_plot_1d_field_input_validation(coil):
    """Test input validation for plot_1d_field."""
    with pytest.raises(ValueError, match="field_component must be"):
        plot_1d_field(coil, field_component="invalid")


# --- Tests for plot_2d_field ---
@pytest.mark.parametrize("field_component", ["x", "y", "z", "norm"])
def test_plot_2d_field_heatmap_runs_successfully(coil, field_component):
    """Test that plot_2d_field runs without errors for heatmap configurations."""
    try:
        plot_2d_field(
            coil,
            field_component=field_component,
            plot_type="heatmap",
            plane="xy",
            num_points_a=2,  # Drastically reduced for speed
            num_points_b=2,  # Drastically reduced for speed
        )
    except Exception as e:
        pytest.fail(f"plot_2d_field (heatmap) raised an unexpected exception: {e}")


@pytest.mark.parametrize("plot_type", ["quiver", "streamline"])
def test_plot_2d_field_vector_runs_successfully(coil, plot_type):
    """Test that plot_2d_field runs without errors for vector plot configurations."""
    try:
        plot_2d_field(
            coil,
            field_component="norm",  # Component ignored for vector plots
            plot_type=plot_type,
            plane="xy",
            num_points_a=2,  # Drastically reduced for speed
            num_points_b=2,  # Drastically reduced for speed
        )
    except Exception as e:
        pytest.fail(f"plot_2d_field ({plot_type}) raised an unexpected exception: {e}")


def test_plot_2d_field_input_validation(coil):
    """Test input validation for plot_2d_field."""
    with pytest.raises(ValueError, match="plot_type must be"):
        plot_2d_field(coil, plot_type="invalid")


# --- Tests for plot_field_vectors_3d ---
def test_plot_field_vectors_3d_runs_successfully(coil):
    """Test that plot_field_vectors_3d runs without errors."""
    try:
        plot_field_vectors_3d(coil, num_points_a=2, num_points_b=2, num_points_c=2)
    except Exception as e:
        pytest.fail(f"plot_field_vectors_3d raised an unexpected exception: {e}")
