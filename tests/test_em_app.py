# Tests for the EMLibrary module
import pytest
import numpy as np
import mtflib
from em_app import biot_savart
from em_app import currentcoils
from mtflib import MultivariateTaylorFunction

@pytest.fixture(scope="function", autouse=True)
def setup_function():
    """Initialize MTF globals for each test in this module."""
    MultivariateTaylorFunction.initialize_mtf(max_order=5, max_dimension=4)
    MultivariateTaylorFunction.set_etol(1e-12)
    yield
    mtflib.taylor_function.MultivariateTaylorFunction._INITIALIZED = False


def test_ring_loop_biot_savart():
    """
    Test the Biot-Savart calculation for a single circular loop.
    Compares the calculated B-field at the center with the analytical formula.
    B = (mu_0 * I) / (2 * R), assuming I=1A
    """
    ring_radius = 0.1  # meters
    num_segments = 100

    pose = currentcoils.Pose(
        position=np.array([0, 0, 0]),
        orientation_axis=np.array([0, 0, 1]),
        orientation_angle=0
    )

    ring = currentcoils.RingLoop(
        current=1.0,
        radius=ring_radius,
        num_segments=num_segments,
        pose=pose
    )

    field_points = np.array([[0, 0, 0]])
    b_field_numerical = ring.biot_savart(field_points)

    # Analytical solution for B-field at the center of a loop
    # B = mu_0 * I / (2 * R), with I=1
    # mu_0 = 4 * pi * 1e-7
    # B = (4 * np.pi * 1e-7) / (2 * ring_radius) in the z-direction
    b_field_analytical_z = (4 * np.pi * 1e-7) / (2 * ring_radius)

    assert np.allclose(b_field_numerical[0, 0], 0)
    assert np.allclose(b_field_numerical[0, 1], 0)
    assert np.allclose(b_field_numerical[0, 2], b_field_analytical_z, rtol=1e-3)


def test_ring_loop_segment_positions():
    """
    Test that the generated segment centers lie on a circle of the correct radius.
    """
    ring_radius = 0.1
    num_segments = 10

    pose = currentcoils.Pose(
        position=np.array([0, 0, 0]),
        orientation_axis=np.array([0, 0, 1]),
        orientation_angle=0
    )

    ring = currentcoils.RingLoop(
        current=1.0,
        radius=ring_radius,
        num_segments=num_segments,
        pose=pose
    )

    segments = ring.get_segments()
    positions = segments[:, 0:3]

    for pos in positions:
        distance_from_center = np.linalg.norm(pos - pose.position)
        assert np.isclose(distance_from_center, ring_radius)

def test_rectangular_loop_biot_savart():
    """
    Test the Biot-Savart calculation for a rectangular loop.
    This is a qualitative test to ensure it runs without errors.
    A more quantitative test would require a known analytical solution.
    """
    pose = currentcoils.Pose(
        position=np.array([0, 0, 0]),
        orientation_axis=np.array([0, 0, 1]),
        orientation_angle=0
    )

    rect_loop = currentcoils.RectangularLoop(
        current=1.0,
        width=0.2,
        height=0.4,
        num_segments_per_side=10,
        pose=pose
    )

    field_points = np.array([[0, 0, 0.1]])
    b_field = rect_loop.biot_savart(field_points)

    # Expect a non-zero field
    assert not np.allclose(b_field, 0)
