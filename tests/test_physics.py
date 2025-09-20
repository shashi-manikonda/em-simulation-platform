import numpy as np
import pytest
from em_app.currentcoils import RectangularCoil, RingCoil, Coil, StraightWire
from em_app.biot_savart import mu_0_4pi, serial_biot_savart

class TestCoil(Coil):
    def __init__(self, current, element_center, element_length, element_direction):
        super().__init__(current)
        self.segment_centers = np.array([element_center])
        self.segment_lengths = np.array([element_length])
        self.segment_directions = np.array([element_direction])

def test_single_straight_segment():
    """
    Test the magnetic field of a single straight wire segment with direct input.
    """
    current = 1.0
    L = 2.0
    element_center = np.array([0, 0, 0])
    element_length = L / 100
    element_direction = np.array([1, 0, 0])

    coil = StraightWire(current, np.array([-L/2, 0, 0]), np.array([L/2, 0, 0]), 100)

    d = 0.5
    field_point = np.array([[0, d, 0]])

    b_vector = coil.biot_savart(field_point)[0]
    b_numerical = b_vector.to_numpy_array()

    # Analytical solution for a finite straight wire
    mu_0 = mu_0_4pi * 4 * np.pi
    b_analytical_z = (mu_0 * current * L) / (2 * np.pi * d * np.sqrt(L**2 + 4 * d**2))

    # The field should be in the +z direction
    assert np.isclose(b_numerical[2], b_analytical_z, rtol=1e-3)

def test_long_straight_wire():
    """
    Test the magnetic field of a long, straight wire.
    """
    current = 2.0
    p1 = np.array([-100, -0.1, 0])
    p2 = np.array([100, -0.1, 0])
    p4 = np.array([-100, 0.1, 0])
    num_segments_per_side = 1000

    coil = RectangularCoil(current, p1, p2, p4, num_segments_per_side)

    d = 0.5
    field_point = np.array([[0, d, 0]])

    b_vector = coil.biot_savart(field_point)[0]
    b_numerical = b_vector.to_numpy_array()

    mu_0 = mu_0_4pi * 4 * np.pi
    # The analytical solution for a rectangular loop is complex.
    # I will approximate it as two infinite wires, which should be accurate for this geometry.
    b_z_analytical = (mu_0 * current / (2 * np.pi)) * (1 / (d - 0.1) + 1 / (d + 0.1))

    assert np.allclose(b_numerical[2], b_z_analytical, rtol=1e-2)
    assert np.allclose(b_numerical[0], 0, atol=1e-9)
    assert np.allclose(b_numerical[1], 0, atol=1e-9)

def test_circular_loop_center():
    """
    Test the magnetic field at the center of a circular loop.
    """
    radius = 0.5
    current = 1.5
    num_segments = 50
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])

    coil = RingCoil(current, radius, num_segments, center, axis)
    field_point = np.array([[0, 0, 0]])

    b_vector = coil.biot_savart(field_point)[0]
    b_numerical = b_vector.to_numpy_array()

    mu_0 = mu_0_4pi * 4 * np.pi
    b_analytical_z = (mu_0 * current) / (2 * radius)

    assert np.allclose(b_numerical[2], b_analytical_z, rtol=1e-3)
    assert np.allclose(b_numerical[0], 0, atol=1e-9)
    assert np.allclose(b_numerical[1], 0, atol=1e-9)

def test_divergence_of_b():
    """
    Test that the divergence of the magnetic field is zero.
    """
    radius = 0.5
    current = 1.5
    num_segments = 20
    center = np.array([0, 0, 0])
    axis = np.array([0, 0, 1])

    coil = RingCoil(current, radius, num_segments, center, axis)

    from mtflib import mtf
    field_point = np.array([[mtf.var(1), mtf.var(2), mtf.var(3)]])

    b_vector = coil.biot_savart(field_point)[0]
    divergence = b_vector.divergence()

    assert np.isclose(divergence.extract_coefficient(tuple([0] * divergence.dimension)).item(), 0, atol=1e-9)
