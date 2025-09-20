import numpy as np
import warnings
import math
import matplotlib.pyplot as plt

# Import the mtflib and biot_savart libraries
from mtflib import mtf
from .biot_savart import serial_biot_savart, mpi_biot_savart, mpi_installed
from .magneticfield import Bvec


def _rotation_matrix(axis, angle):
    """
    (PRIVATE) Generates a rotation matrix about an arbitrary axis using
    quaternion parameters.

    Args:
        axis (np.ndarray): The axis of rotation.
        angle (float): The angle of rotation in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2)
    b, c, d = -axis * math.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, bd, cd = b * c, b * d, c * d
    ad, ac, ab = a * d, a * c, a * b
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def _rotation_matrix_align_vectors(v1, v2):
    """
    (PRIVATE) Generates a rotation matrix to rotate vector v1 to align
    with vector v2.

    Args:
        v1 (np.ndarray): The starting vector.
        v2 (np.ndarray): The target vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    v_cross = np.cross(v1_u, v2_u)
    if np.allclose(v_cross, 0):
        if np.dot(v1_u, v2_u) < 0:
            return _rotation_matrix(np.array([1, 0, 0]), np.pi)
        return np.eye(3)

    rotation_axis = v_cross / np.linalg.norm(v_cross)
    rotation_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return _rotation_matrix(rotation_axis, rotation_angle)


def _current_ring(
    ring_radius,
    num_segments_ring,
    ring_center_point,
    ring_axis_direction,
    return_mtf=True,
):
    """
    (PRIVATE) Generates MTF representations for segments of a current ring defined
    by its center point and axis direction.

    This is a private helper function and should not be used directly.

    Args:
        ring_radius (float): Radius of the current ring.
        num_segments_ring (int): Number of segments to discretize the ring into.
        ring_center_point (numpy.ndarray): (3,) array defining the center
                                           coordinates (x, y, z) of the ring.
        ring_axis_direction (numpy.ndarray): (3,) array defining the direction
                                             vector of the ring's axis
                                             (normal to the plane of the ring).
        return_mtf (bool): If True, returns MTF objects for use within `mtflib`.
                           If False, returns raw NumPy arrays for external use.

    Returns:
        tuple: A tuple containing:
            - segment_representations (numpy.ndarray): (N,) array of MTFs or
                                                       (N, 3) array of segment
                                                       center points.
            - element_lengths_ring (numpy.ndarray): (N,) array of lengths of
                                                    each ring segment (dl).
            - direction_vectors (numpy.ndarray): (N, 3) array of MTF direction
                                                 vectors or NumPy direction
                                                 vectors.
    """
    d_phi = 2 * np.pi / num_segments_ring
    ring_axis_direction_unit = ring_axis_direction / np.linalg.norm(ring_axis_direction)

    rotation_align_z_axis = _rotation_matrix_align_vectors(
        np.array([0, 0, 1.0]), ring_axis_direction_unit
    )

    if return_mtf:
        u = mtf.var(4)
        segment_mtfs_ring = []
        element_lengths_ring = []
        direction_vectors_ring = []

        ring_center_point_mtf = mtf.from_numpy_array(ring_center_point)

        for i in range(num_segments_ring):
            phi = (i + 0.5 + 0.5 * u) * d_phi
            x_center = ring_radius * mtf.cos(phi)
            y_center = ring_radius * mtf.sin(phi)
            z_center = mtf.from_constant(0.0)

            center_point = np.array([x_center, y_center, z_center], dtype=object)
            center_point_rotated = np.dot(rotation_align_z_axis, center_point)
            center_point_translated = center_point_rotated + ring_center_point_mtf
            segment_mtfs_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            direction_base = np.array(
                [-mtf.sin(phi), mtf.cos(phi), mtf.from_constant(0.0)], dtype=object
            )
            direction_rotated = np.dot(rotation_align_z_axis, direction_base)
            norm_mtf_squared = (
                direction_rotated[0] ** 2
                + direction_rotated[1] ** 2
                + direction_rotated[2] ** 2
            )
            norm_mtf_squared.set_coefficient((0, 0, 0, 0), 1.0)
            norm_mtf = mtf.sqrt(norm_mtf_squared)
            direction_normalized_mtf = [
                direction_rotated[i] / norm_mtf for i in range(3)
            ]
            direction_vectors_ring.append(direction_normalized_mtf)

        return (
            np.array(segment_mtfs_ring, dtype=object),
            np.array(element_lengths_ring),
            np.array(direction_vectors_ring, dtype=object),
        )

    else:  # return_mtf is False, return NumPy arrays
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
            center_point_rotated = rotation_align_z_axis @ np.array(
                [x_center, y_center, z_center]
            )
            center_point_translated = center_point_rotated + ring_center_point
            segment_centers_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            # Tangent direction at center point (for base ring in xy-plane):
            direction_base = np.array([-np.sin(phi), np.cos(phi), 0])
            direction_rotated = rotation_align_z_axis @ direction_base
            direction_normalized = direction_rotated / np.linalg.norm(direction_rotated)
            direction_vectors_ring.append(direction_normalized)

        return (
            np.array(segment_centers_ring),
            np.array(element_lengths_ring),
            np.array(direction_vectors_ring),
        )


def _straight_wire_segments(start_point, end_point, num_segments):
    """
    (PRIVATE) Helper to discretize a straight wire into segments.

    This is a private helper function and should not be used directly.
    It generates segment data as MTF objects for a straight wire.

    Args:
        start_point (np.ndarray): The starting point of the wire.
        end_point (np.ndarray): The ending point of the wire.
        num_segments (int): Number of segments.

    Returns:
        tuple: A tuple of segment centers, lengths, and directions as MTF objects.
    """
    start_point_mtf = mtf.from_numpy_array(start_point)
    end_point_mtf = mtf.from_numpy_array(end_point)

    wire_vector_mtf = end_point_mtf - start_point_mtf
    wire_length_mtf = mtf.sqrt(mtf.sum(wire_vector_mtf**2))

    segment_length_mtf = wire_length_mtf / num_segments
    wire_direction_mtf = wire_vector_mtf / wire_length_mtf

    segment_centers = np.empty((num_segments, 3), dtype=object)

    for i in range(num_segments):
        factor = (i + 0.5) * segment_length_mtf
        segment_centers[i] = start_point_mtf + wire_direction_mtf * factor

    segment_lengths = np.full(num_segments, segment_length_mtf)
    segment_directions = np.full((num_segments, 3), wire_direction_mtf, dtype=object)

    return segment_centers, segment_lengths, segment_directions

def _get_3d_axes(ax=None):
    """
    Helper to get a 3D matplotlib axis. Creates a new one if not provided.

    Args:
        ax (matplotlib.axes.Axes3D): The axis to use.

    Returns:
        matplotlib.axes.Axes3D: The 3D axis.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    return ax

class Coil(object):
    """
    Base class for a current-carrying coil.

    This class provides a common interface for different coil shapes,
    storing their properties (like current) and the discretized segments
    used for numerical calculations.
    """

    def __init__(self, current):
        """
        Initializes the base Coil with a current value.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current
                flowing through the coil.
        """
        self.current = mtf.to_mtf(current)

        # These will be populated by subclasses
        self.segment_centers = None
        self.segment_lengths = None
        self.segment_directions = None

    def get_segments(self):
        """
        Returns the segments of the coil.

        Returns:
            tuple: A tuple containing:
                - segment_centers (np.ndarray): Array of MTF center points.
                - segment_lengths (np.ndarray): Array of MTF segment lengths.
                - segment_directions (np.ndarray): Array of MTF direction vectors.
        """
        if self.segment_centers is None:
            raise NotImplementedError("Subclass must implement segment generation.")
        return self.segment_centers, self.segment_lengths, self.segment_directions

    def get_max_size(self) -> np.ndarray:
        """
        Calculates the maximum extent of the coil in each dimension.

        Returns:
            np.ndarray: A (3,) array of the maximum size of the coil
                        (width, height, depth).
        """
        if self.segment_centers is None:
            return np.zeros(3)

        # Convert MTF objects to NumPy arrays for calculation
        centers_numerical = mtf.to_numpy_array(self.segment_centers)
        directions_numerical = mtf.to_numpy_array(self.segment_directions)

        # Calculate max and min coordinates
        all_coords = np.vstack(
            [
                centers_numerical,
                centers_numerical
                + (directions_numerical * self.segment_lengths.reshape(-1, 1)),
                centers_numerical
                - (directions_numerical * self.segment_lengths.reshape(-1, 1)),
            ]
        )

        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)

        return max_coords - min_coords

    def plot(self, ax=None, color="b"):
        """
        Plots the coil segments in a 3D matplotlib axis.

        If a 3D axis is provided, the coil will be plotted on it. Otherwise,
        a new figure and a new 3D axis will be created.

        Args:
            ax (matplotlib.axes.Axes3D): The 3D axis to plot on. Defaults to None.
            color (str): The color for the plotted segments. Defaults to "b".
        """
        if self.segment_centers is None:
            warnings.warn("No segments to plot.")
            return

        # Get or create the 3D axes
        plot_ax = _get_3d_axes(ax)

        centers = mtf.to_numpy_array(self.segment_centers)
        directions = mtf.to_numpy_array(self.segment_directions)
        lengths = self.segment_lengths

        for i in range(len(centers)):
            start_point = centers[i] - directions[i] * lengths[i] / 2
            end_point = centers[i] + directions[i] * lengths[i] / 2
            x_vals = [start_point[0], end_point[0]]
            y_vals = [start_point[1], end_point[1]]
            z_vals = [start_point[2], end_point[2]]
            plot_ax.plot(x_vals, y_vals, z_vals, color=color)
            
        # Ensure the plot is rendered if a new figure was created
        if ax is None:
            plt.show()
    
    def biot_savart(self, field_points, backend="python"):
        """
        Calculates the magnetic field generated by the coil at a set of
        field points using the Biot-Savart law.

        Args:
            field_points (np.ndarray or np.ndarray of
                          mtf.MultivariateTaylorFunction):
                The points (x, y, z) where the magnetic field should be
                calculated. Can be a (M, 3) NumPy array of numbers or MTF objects.
            backend (str, optional): The backend to use for the calculation.
                Options include 'python', 'cpp', 'c', 'cpp_v2', and 'mpi'.
                Defaults to 'python'. Note that 'mpi' requires the 'mpi4py'
                library.

        Returns:
            np.ndarray: A (M,) array of Bvec objects, each representing the
                        magnetic field vector at a corresponding field point.
        """
        # --- Check the calling child class ---
        calling_class_name = self.__class__.__name__
        print(f"Biot-Savart method called by: {calling_class_name}")

        # Check if segments have been generated
        if self.segment_centers is None:
            raise RuntimeError("Coil segments have not been generated.")

        # The Biot-Savart formula is a sum over all segments.
        # B = (mu_0 / 4pi) * sum( (I * dl x r) / |r|^3 )
        # Here, we use the `serial_biot_savart` function to perform the calculation.

        if backend == "mpi":
            if not mpi_installed:
                raise ImportError(
                    "The 'mpi' backend requires the 'mpi4py' library to be installed."
                )
            b_field_vectors = mpi_biot_savart(
                element_centers=self.segment_centers,
                element_lengths=self.segment_lengths,
                element_directions=self.segment_directions,
                field_points=field_points,
            )
        else:
            b_field_vectors = serial_biot_savart(
                element_centers=self.segment_centers,
                element_lengths=self.segment_lengths,
                element_directions=self.segment_directions,
                field_points=field_points,
                backend=backend,
            )

        # Apply the current scaling
        if isinstance(self.current, mtf):
            b_field_vectors = np.array([Bvec(self.current * vec[0], self.current * vec[1], self.current * vec[2]) for vec in b_field_vectors], dtype=object)
        else:
            b_field_vectors *= self.current
        
        b_vectors_numerical = np.array([b.to_numpy_array() for b in b_field_vectors])

        if calling_class_name == "RingCoil":
            for i, vec in enumerate(b_field_vectors):
                vec.Bx = vec.Bx.integrate(4, -1, 1)
                vec.By = vec.By.integrate(4, -1, 1)
                vec.Bz = vec.Bz.integrate(4, -1, 1)
                b_field_vectors[i] = vec 
        return b_field_vectors


class RingCoil(Coil):
    """
    Represents a circular current-carrying coil.
    """

    def __init__(self, current, radius, num_segments, center_point, axis_direction):
        """
        Initializes a circular coil.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current.
            radius (float): Radius of the coil.
            num_segments (int): Number of segments for discretization.
            center_point (np.ndarray): (3,) array for the center coordinates.
            axis_direction (np.ndarray): (3,) array for the axis direction.
        """
        super().__init__(current)

        self.radius = radius
        self.num_segments = num_segments
        self.center_point = center_point
        self.axis_direction = axis_direction

        # Generate the segments using the helper function
        self.segment_centers, self.segment_lengths, self.segment_directions = (
            _current_ring(
                radius, num_segments, center_point, axis_direction, return_mtf=True
            )
        )


class RectangularCoil(Coil):
    """
    Represents a rectangular current-carrying coil.
    """

    def __init__(self, current, p1, p2, p4, num_segments_per_side):
        """
        Initializes a rectangular coil.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current.
            p1 (np.ndarray): First corner of the rectangle.
            p2 (np.ndarray): Second corner, defining the first side from p1.
            p4 (np.ndarray): Fourth corner, defining the second side from p1.
            num_segments_per_side (int): Segments per side.
        """
        super().__init__(current)

        p1 = np.array(p1)
        p2 = np.array(p2)
        p4 = np.array(p4)

        if not (np.isclose(np.dot(p2 - p1, p4 - p1), 0)):
            raise ValueError("Side vectors from p1 must be orthogonal.")

        p3 = p2 + (p4 - p1)
        corners = [p1, p2, p3, p4]

        # Use the straight wire logic to generate segments for each side
        all_segments = []
        for i in range(4):
            start_p = corners[i]
            end_p = corners[(i + 1) % 4]
            all_segments.append(
                _straight_wire_segments(start_p, end_p, num_segments_per_side)
            )

        # Concatenate segments from all four sides
        self.segment_centers = np.concatenate([s[0] for s in all_segments], axis=0)
        self.segment_lengths = np.concatenate([s[1] for s in all_segments])
        self.segment_directions = np.concatenate([s[2] for s in all_segments], axis=0)