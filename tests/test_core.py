import numpy as np
import pytest
from em_app.biot_savart import _python_biot_savart_core, mu_0_4pi

def test_core_biot_savart():
    """
    Test the core Biot-Savart calculation with simple inputs.
    """
    source_points = np.array([[0, 0, 0]])
    dl_vectors = np.array([[0.1, 0, 0]])
    field_points = np.array([[0, 1, 0]])
    current = 1.0

    b_field = _python_biot_savart_core(source_points, dl_vectors, field_points, current=current)

    # Analytical solution
    # dB = (mu_0 * I / 4pi) * (dl x r) / |r|^3
    # dl = [0.1, 0, 0], r = [0, 1, 0], |r| = 1
    # dl x r = [0, 0, 0.1]
    # dB_z = (1e-7 * 1.0) * 0.1 / 1^3 = 1e-8
    b_analytical_z = 1e-8

    assert np.isclose(b_field[0, 2], b_analytical_z)
