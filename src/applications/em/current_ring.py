import numpy as np
from mtflib import MultivariateTaylorFunction, Var, convert_to_mtf
from mtflib.elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor

def current_ring(ring_radius, num_segments_ring, ring_center_point, ring_axis_direction, return_mtf=True):
    """
    Generates MTF representations for segments of a current ring defined by its center point
    and axis direction.

    Args:
        ring_radius (float): Radius of the current ring.
        num_segments_ring (int): Number of segments to discretize the ring into.
        ring_center_point (numpy.ndarray): (3,) array defining the center coordinates (x, y, z) of the ring.
        ring_axis_direction (numpy.ndarray): (3,) array defining the direction vector of the ring's axis
                                            (normal to the plane of the ring).
        return_mtf (bool): If True, returns MTF objects for use within `mtflib`.
                           If False, returns raw NumPy arrays for external use.

    Returns:
        tuple: A tuple containing:
            - segment_representations (numpy.ndarray): (N,) array of MTFs or (N, 3) array of segment center points.
            - element_lengths_ring (numpy.ndarray): (N,) array of lengths of each ring segment (dl).
            - direction_vectors (numpy.ndarray): (N, 3) array of MTF direction vectors or NumPy direction vectors.
    """
    d_phi = 2 * np.pi / num_segments_ring
    ring_axis_direction_unit = ring_axis_direction / np.linalg.norm(ring_axis_direction)

    # Rotation matrix to align z-axis with ring_axis_direction
    def rotation_matrix_align_vectors(v1, v2):
        """Generates rotation matrix to rotate vector v1 to align with v2."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v_cross = np.cross(v1_u, v2_u)
        if np.allclose(v_cross, 0):
            if np.dot(v1_u, v2_u) < 0:
                return rotation_matrix(np.array([1, 0, 0]), np.pi)
            else:
                return np.eye(3)

        rotation_axis = v_cross / np.linalg.norm(v_cross)
        rotation_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        return rotation_matrix(rotation_axis, rotation_angle)

    def rotation_matrix(axis, angle):
        """Rotation matrix about arbitrary axis using quaternion parameters."""
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle/2)
        b,c,d = -axis*np.sin(angle/2)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, bd, cd = b*c, b*d, c*d
        ad, ac, ab = a*d, a*c, a*b
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    rotation_align_z_axis = rotation_matrix_align_vectors(np.array([0, 0, 1.0]), ring_axis_direction_unit)

    if return_mtf:
        u = Var(4)
        segment_mtfs_ring = []
        element_lengths_ring = []
        direction_vectors_ring = []

        for i in range(num_segments_ring):
            phi = (i + 0.5 + 0.5*u) * d_phi
            x_center = ring_radius * cos_taylor(phi)
            y_center = ring_radius * sin_taylor(phi)
            z_center = convert_to_mtf(0.0)

            center_point = np.array([x_center, y_center, z_center], dtype=object)
            center_point_rotated = np.dot(rotation_align_z_axis, center_point)
            center_point_translated = center_point_rotated + ring_center_point
            segment_mtfs_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            direction_base = np.array([-sin_taylor(phi), cos_taylor(phi), convert_to_mtf(0.0)], dtype=object)
            direction_rotated = np.dot(rotation_align_z_axis, direction_base)
            norm_mtf_squared = direction_rotated[0]**2 + direction_rotated[1]**2 + direction_rotated[2]**2
            norm_mtf_squared.set_coefficient((0,0,0,0), 1.0)
            norm_mtf = sqrt_taylor(norm_mtf_squared)
            direction_normalized_mtf = [direction_rotated[i] / norm_mtf for i in range(3)]
            direction_vectors_ring.append(direction_normalized_mtf)

        return np.array(segment_mtfs_ring, dtype=object), np.array(element_lengths_ring), np.array(direction_vectors_ring, dtype=object)

    else: # return_mtf is False, return NumPy arrays
        segment_centers_ring = []
        element_lengths_ring = []
        direction_vectors_ring = []

        for i in range(num_segments_ring):
            phi = (i + 0.5) * d_phi
            # Base ring in xy-plane
            x_center = ring_radius * np.cos(phi)
            y_center = ring_radius * np.sin(phi)
            z_center = 0.0

            # Rotate and translate center point
            center_point_rotated = rotation_align_z_axis @ np.array([x_center, y_center, z_center])
            center_point_translated = center_point_rotated + ring_center_point
            segment_centers_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            # Tangent direction at center point (for base ring in xy-plane):
            direction_base = np.array([-np.sin(phi), np.cos(phi), 0])
            direction_rotated = rotation_align_z_axis @ direction_base
            direction_normalized = direction_rotated / np.linalg.norm(direction_rotated)
            direction_vectors_ring.append(direction_normalized)

        return np.array(segment_centers_ring), np.array(element_lengths_ring), np.array(direction_vectors_ring)
