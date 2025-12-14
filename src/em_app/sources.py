import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mtflib import mtf

from .vector_fields import Vector


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
    # Input validation
    if not isinstance(axis, np.ndarray) or axis.shape != (3,):
        raise TypeError("Axis must be a 3-element NumPy array.")
    if not isinstance(angle, (int, float)):
        raise TypeError("Angle must be a number.")

    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2)
    b, c, d = -axis * math.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, bd, cd = b * c, b * d, c * d
    ad, ac, ab = a * d, a * c, a * b
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ])


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
    # Input validation
    if not isinstance(v1, np.ndarray) or v1.shape != (3,):
        raise TypeError("v1 must be a 3-element NumPy array.")
    if not isinstance(v2, np.ndarray) or v2.shape != (3,):
        raise TypeError("v2 must be a 3-element NumPy array.")

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

    def __init__(self, current, use_mtf_for_segments=True, wire_thickness=0.001):
        """
        Initializes the base Coil with a current value.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current
                flowing through the coil.
            use_mtf_for_segments (bool): Whether to use MTF for segments.
            wire_thickness (float): The thickness of the wire in meters.
        """
        # Input validation
        if not isinstance(use_mtf_for_segments, bool):
            raise TypeError("use_mtf_for_segments must be a boolean.")

        self.current = mtf.to_mtf(current)
        self.use_mtf_for_segments = use_mtf_for_segments
        self.wire_thickness = wire_thickness

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
        centers_numerical = np.array([c.to_numpy_array() for c in self.segment_centers])
        directions_numerical = np.array([
            d.to_numpy_array() for d in self.segment_directions
        ])

        # Calculate max and min coordinates
        all_coords = np.vstack([
            centers_numerical,
            centers_numerical
            + (directions_numerical * self.segment_lengths.reshape(-1, 1)),
            centers_numerical
            - (directions_numerical * self.segment_lengths.reshape(-1, 1)),
        ])

        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)

        return max_coords - min_coords

    def get_center_point(self) -> np.ndarray:
        """
        Calculates the approximate center point of the coil.

        Returns:
            np.ndarray: A (3,) array representing the center of the coil.
        """
        if self.segment_centers is None:
            return np.zeros(3)

        if isinstance(self.segment_centers[0], Vector):
            centers_numerical = np.array([
                c.to_numpy_array() for c in self.segment_centers
            ])
        else:
            centers_numerical = self.segment_centers

        return np.mean(centers_numerical, axis=0)

    def plot(
        self,
        ax=None,
        color="#B87333",
        num_interpolation_points=2,
        wire_thickness=None,
        show_axis=False,
    ):
        """
        Plots the coil segments in a 3D matplotlib axis.

        If a 3D axis is provided, the coil will be plotted on it. Otherwise,
        a new figure and a new 3D axis will be created.

        Args:
            ax (matplotlib.axes.Axes3D): The 3D axis to plot on. Defaults to
                None.
            color (str): The color for the plotted segments. Defaults to a
                copper-like hex code.
            num_interpolation_points (int): The number of points to plot for
                each segment, including start and end points. This is only
                used when `use_mtf_for_segments` is `True`. Defaults to 2.
            wire_thickness (float): The thickness of the wire to plot. Defaults
                to the thickness specified at initialization.
            show_axis (bool): Whether to plot the central axis of the coil.
                Defaults to False.
        """
        if self.segment_centers is None:
            warnings.warn("No segments to plot.")
            return

        plot_ax = _get_3d_axes(ax)

        # Determine the line width from the wire thickness
        line_width = 1.0
        if wire_thickness is None:
            if self.wire_thickness is not None:
                line_width = self.wire_thickness * 1000  # Heuristic scaling
        else:
            line_width = wire_thickness * 1000  # Heuristic scaling

        # Plot with interpolation if use_mtf_for_segments is True and more
        # than 2 points are requested
        if self.use_mtf_for_segments and num_interpolation_points > 2:
            u_points = np.zeros((num_interpolation_points, 4))
            u_points[:, 3] = np.linspace(-1, 1, num_interpolation_points)

            for center_vec_mtf in self.segment_centers:
                # Evaluate the MTF along the segment parameter u
                # Assuming the MTF for the center point has one variable, `u`,
                # at index 4
                evaluated_points = np.array([
                    x.neval(u_points) for x in center_vec_mtf
                ]).T
                plot_ax.plot(
                    evaluated_points[:, 0],
                    evaluated_points[:, 1],
                    evaluated_points[:, 2],
                    color=color,
                    linewidth=line_width,
                )
        else:
            if not self.use_mtf_for_segments and num_interpolation_points > 2:
                warnings.warn(
                    "Ignoring `num_interpolation_points` because "
                    "`use_mtf_for_segments` is False."
                )
            # Fallback to the original behavior
            centers = np.array([c.to_numpy_array() for c in self.segment_centers])
            directions = np.array([d.to_numpy_array() for d in self.segment_directions])
            lengths = self.segment_lengths
            for i in range(len(centers)):
                start_point = centers[i] - directions[i] * lengths[i] / 2
                end_point = centers[i] + directions[i] * lengths[i] / 2
                x_vals = [start_point[0], end_point[0]]
                y_vals = [start_point[1], end_point[1]]
                z_vals = [start_point[2], end_point[2]]
                plot_ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=line_width)

        # Add a new section to plot the coil axis if requested
        if show_axis:
            # Check for a specific coil type that has a defined axis
            if isinstance(self, RingCoil):
                # Use the coil's properties to define the axis
                center = self.center_point
                direction = self.axis_direction / np.linalg.norm(self.axis_direction)

                # Determine the length of the axis to be plotted
                axis_length = 2.5 * self.radius

                start_axis = center - axis_length * direction
                end_axis = center + axis_length * direction

                x_vals = [start_axis[0], end_axis[0]]
                y_vals = [start_axis[1], end_axis[1]]
                z_vals = [start_axis[2], end_axis[2]]

                plot_ax.plot(
                    x_vals, y_vals, z_vals, color="gray", linestyle="--", linewidth=0.8
                )

            elif isinstance(self, RectangularCoil):
                # Calculate the normal to the plane of the rectangle
                p1, p2, p4 = self.p1, self.p2, self.p4
                vec1 = p2 - p1
                vec2 = p4 - p1
                normal = np.cross(vec1, vec2)
                normal = normal / np.linalg.norm(normal)

                # Calculate the center of the rectangle
                center = (p1 + (p2 + p4 - p1)) / 2

                # Determine axis length based on the largest side
                side1_length = np.linalg.norm(vec1)
                side2_length = np.linalg.norm(vec2)
                axis_length = 1.5 * max(side1_length, side2_length)

                start_axis = center - axis_length * normal
                end_axis = center + axis_length * normal

                x_vals = [start_axis[0], end_axis[0]]
                y_vals = [start_axis[1], end_axis[1]]
                z_vals = [start_axis[2], end_axis[2]]

                plot_ax.plot(
                    x_vals, y_vals, z_vals, color="gray", linestyle="--", linewidth=0.8
                )
            else:
                warnings.warn(
                    f"Coil type {type(self).__name__} does not have a "
                    "well-defined axis for plotting."
                )

        if ax is None:
            plt.show()


class RingCoil(Coil):
    """
    Represents a circular current-carrying coil.
    """

    def __init__(
        self,
        current,
        radius,
        num_segments,
        center_point,
        axis_direction,
        use_mtf_for_segments=True,
        wire_thickness=0.001,
    ):
        """
        Initializes a circular coil.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current.
            radius (float): Radius of the coil.
            num_segments (int): Number of segments for discretization.
            center_point (np.ndarray): (3,) array for the center coordinates.
            axis_direction (np.ndarray): (3,) array for the axis direction.
        """
        # Input validation
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number.")
        if not isinstance(num_segments, int) or num_segments <= 0:
            raise ValueError("Number of segments must be a positive integer.")
        if not isinstance(center_point, (np.ndarray, list)) or len(center_point) != 3:
            raise ValueError("Center point must be a 3-element list or NumPy array.")
        if (
            not isinstance(axis_direction, (np.ndarray, list))
            or len(axis_direction) != 3
        ):
            raise ValueError("Axis direction must be a 3-element list or NumPy array.")
        if np.linalg.norm(axis_direction) == 0:
            raise ValueError("Axis direction vector cannot be a zero vector.")

        super().__init__(current, use_mtf_for_segments, wire_thickness)

        self.radius = radius
        self.num_segments = num_segments
        self.center_point = center_point
        self.axis_direction = axis_direction

        # Generate the segments using the helper function
        self.segment_centers, self.segment_lengths, self.segment_directions = (
            self.generate_geometry(
                radius,
                num_segments,
                center_point,
                axis_direction,
                use_mtf_for_segments,
            )
        )

    @staticmethod
    def generate_geometry(
        ring_radius,
        num_segments_ring,
        ring_center_point,
        ring_axis_direction,
        use_mtf_for_segments=True,
    ):
        """
        (PRIVATE) Generates MTF representations for segments of a current ring.

        This is a private helper function and should not be used directly.

        Args:
            ring_radius (float): Radius of the current ring.
            num_segments_ring (int): Number of segments to discretize the ring
                into.
            ring_center_point (numpy.ndarray): (3,) array defining the center
                coordinates (x, y, z) of the ring.
            ring_axis_direction (numpy.ndarray): (3,) array defining the
                direction vector of the ring's axis (normal to the plane of
                the ring).

        Returns:
            tuple: A tuple containing:
                - segment_representations (numpy.ndarray): (N,) array of MTFs
                  or (N, 3) array of segment center points.
                - element_lengths_ring (numpy.ndarray): (N,) array of lengths
                  of each ring segment (dl).
                - direction_vectors (numpy.ndarray): (N, 3) array of MTF
                  direction vectors or NumPy direction vectors.
        """
        d_phi = 2 * np.pi / num_segments_ring
        ring_axis_direction_unit = ring_axis_direction / np.linalg.norm(
            ring_axis_direction
        )

        rotation_align_z_axis = _rotation_matrix_align_vectors(
            np.array([0, 0, 1.0]), ring_axis_direction_unit
        )

        if use_mtf_for_segments:
            u = mtf.var(4)  # Use a variable for integration later
        else:
            u = 0.0

        segment_mtfs_ring = []
        element_lengths_ring = []
        direction_vectors_ring = []

        ring_center_point_mtf = np.array([mtf.to_mtf(x) for x in ring_center_point])

        for i in range(num_segments_ring):
            phi = (i + 0.5 + 0.5 * u) * d_phi
            x_center = ring_radius * mtf.cos(phi)
            y_center = ring_radius * mtf.sin(phi)
            z_center = mtf.from_constant(0.0)

            center_point = np.array([x_center, y_center, z_center], dtype=object)
            center_point_rotated = np.dot(rotation_align_z_axis, center_point)
            center_point_translated = center_point_rotated + ring_center_point_mtf
            segment_mtfs_ring.append(Vector(center_point_translated))

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
            direction_vectors_ring.append(Vector(direction_normalized_mtf))

        return (
            np.array(segment_mtfs_ring, dtype=object),
            np.array(element_lengths_ring),
            np.array(direction_vectors_ring, dtype=object),
        )


class RectangularCoil(Coil):
    """
    Represents a rectangular current-carrying coil.
    """

    def __init__(
        self,
        current,
        p1,
        p2,
        p4,
        num_segments_per_side,
        use_mtf_for_segments=True,
        wire_thickness=0.001,
    ):
        """
        Initializes a rectangular coil.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current.
            p1 (np.ndarray): First corner of the rectangle.
            p2 (np.ndarray): Second corner, defining the first side from p1.
            p4 (np.ndarray): Fourth corner, defining the second side from p1.
            num_segments_per_side (int): Segments per side.
        """
        # Input validation
        if not isinstance(p1, (np.ndarray, list)) or len(p1) != 3:
            raise ValueError("p1 must be a 3-element list or NumPy array.")
        if not isinstance(p2, (np.ndarray, list)) or len(p2) != 3:
            raise ValueError("p2 must be a 3-element list or NumPy array.")
        if not isinstance(p4, (np.ndarray, list)) or len(p4) != 3:
            raise ValueError("p4 must be a 3-element list or NumPy array.")
        if not isinstance(num_segments_per_side, int) or num_segments_per_side <= 0:
            raise ValueError("Number of segments per side must be a positive integer.")

        super().__init__(current, use_mtf_for_segments, wire_thickness)
        self.p1 = p1
        self.p2 = p2
        self.p4 = p4
        self.segment_centers, self.segment_lengths, self.segment_directions = (
            self.generate_geometry(
                p1, p2, p4, num_segments_per_side, use_mtf_for_segments
            )
        )

    @staticmethod
    def generate_geometry(p1, p2, p4, num_segments_per_side, use_mtf_for_segments=True):
        """
        (PRIVATE) Generates segments for a rectangular coil.

        Args:
            p1 (np.ndarray): First corner of the rectangle.
            p2 (np.ndarray): Second corner, defining the first side from p1.
            p4 (np.ndarray): Fourth corner, defining the second side from p1.
            num_segments_per_side (int): Segments per side.
            use_mtf_for_segments (bool): Whether to use MTF for segments.

        Returns:
            tuple: A tuple containing:
                - segment_centers (np.ndarray): Array of MTF center points.
                - segment_lengths (np.ndarray): Array of segment lengths.
                - segment_directions (np.ndarray): Array of MTF direction vectors.
        """
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
                StraightWire.generate_geometry(
                    start_p, end_p, num_segments_per_side, use_mtf_for_segments
                )
            )

        # Concatenate segments from all four sides
        segment_centers = np.concatenate([s[0] for s in all_segments], axis=0)
        segment_lengths = np.concatenate([s[1] for s in all_segments])
        segment_directions = np.concatenate([s[2] for s in all_segments], axis=0)

        return segment_centers, segment_lengths, segment_directions


class StraightWire(Coil):
    """
    Represents a single straight current-carrying wire.
    """

    def __init__(
        self,
        current,
        start_point,
        end_point,
        num_segments=1,
        use_mtf_for_segments=True,
        wire_thickness=0.001,
    ):
        """
        Initializes a straight wire.

        Args:
            current (float or mtf.MultivariateTaylorFunction): The current.
            start_point (np.ndarray): The starting point of the wire.
            end_point (np.ndarray): The ending point of the wire.
            num_segments (int): Number of segments. Defaults to 1.
        """
        # Input validation
        if not isinstance(start_point, (np.ndarray, list)) or len(start_point) != 3:
            raise ValueError("Start point must be a 3-element list or NumPy array.")
        if not isinstance(end_point, (np.ndarray, list)) or len(end_point) != 3:
            raise ValueError("End point must be a 3-element list or NumPy array.")
        if np.array_equal(start_point, end_point):
            raise ValueError("Start and end points cannot be the same.")
        if not isinstance(num_segments, int) or num_segments <= 0:
            raise ValueError("Number of segments must be a positive integer.")

        super().__init__(current, use_mtf_for_segments, wire_thickness)
        self.start_point = start_point
        self.end_point = end_point
        (
            self.segment_centers,
            self.segment_lengths,
            self.segment_directions,
        ) = self.generate_geometry(
            start_point, end_point, num_segments, use_mtf_for_segments
        )

    @staticmethod
    def generate_geometry(
        start_point, end_point, num_segments=1, use_mtf_for_segments=True
    ):
        """
        Discretizes the straight wire into segments.
        """
        # Convert start and end points to Vector objects
        start_point_vector = Vector(start_point)
        end_point_vector = Vector(end_point)

        # Calculate the vector representing the entire wire
        wire_vector = end_point_vector - start_point_vector
        wire_length = wire_vector.norm()

        segment_length = wire_length / num_segments
        wire_direction = wire_vector / wire_length

        # Create a linear interpolation of the center points
        num_linspace_points = np.linspace(0.5, num_segments - 0.5, num_segments)

        # Initialize lists to hold the Vector objects for centers and directions
        segment_centers = []
        segment_directions = []

        if use_mtf_for_segments:
            u = mtf.var(4)  # Use a variable for integration later
        else:
            u = 0.0

        # Iterate and create the Vector objects for each segment
        for i in range(num_segments):
            factor = (num_linspace_points[i] + 0.5 * u) * segment_length
            segment_center = start_point_vector + wire_direction * factor

            segment_centers.append(segment_center)
            segment_directions.append(wire_direction)

        # Assign the results to the instance properties
        segment_centers = np.array(segment_centers, dtype=object)
        segment_lengths = np.full(num_segments, segment_length)
        segment_directions = np.array(segment_directions, dtype=object)
        return segment_centers, segment_lengths, segment_directions
