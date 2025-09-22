import numpy as np
import warnings
import math
import matplotlib.pyplot as plt

# Import the mtflib and biot_savart libraries
from mtflib import mtf
from .biot_savart import serial_biot_savart, mpi_biot_savart, mpi_installed
from .magneticfield import Bvec, Vector, Bfield


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
        centers_numerical = np.array(
            [
                np.array([x.get_constant() for x in center_vec])
                for center_vec in self.segment_centers
            ]
        )
        directions_numerical = np.array(
            [
                np.array([d.get_constant() for d in dir_vec])
                for dir_vec in self.segment_directions
            ]
        )

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

    def get_center_point(self) -> np.ndarray:
        """
        Calculates the approximate center point of the coil.

        Returns:
            np.ndarray: A (3,) array representing the center of the coil.
        """
        if self.segment_centers is None:
            return np.zeros(3)
        centers_numerical = np.array(
            [
                np.array([x.get_constant() for x in center_vec])
                for center_vec in self.segment_centers
            ]
        )
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
            ax (matplotlib.axes.Axes3D): The 3D axis to plot on. Defaults to None.
            color (str): The color for the plotted segments. Defaults to a copper-like hex code.
            num_interpolation_points (int): The number of points to plot for each
                                            segment, including start and end points.
                                            This is only used when `use_mtf_for_segments` is `True`.
                                            Defaults to 2.
            wire_thickness (float): The thickness of the wire to plot. Defaults to
                                    the thickness specified at initialization.
            show_axis (bool): Whether to plot the central axis of the coil. Defaults to False.
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

        # Plot with interpolation if use_mtf_for_segments is True and more than 2 points are requested
        if self.use_mtf_for_segments and num_interpolation_points > 2:
            u_points = np.zeros((num_interpolation_points, 4))
            u_points[:, 3] = np.linspace(-1, 1, num_interpolation_points)

            for center_vec_mtf in self.segment_centers:
                # Evaluate the MTF along the segment parameter u
                # Assuming the MTF for the center point has one variable, `u`, at index 4
                evaluated_points = np.array(
                    [x.neval(u_points) for x in center_vec_mtf]
                ).T
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
                    "Ignoring `num_interpolation_points` because `use_mtf_for_segments` is False."
                )
            # Fallback to the original behavior
            centers = np.array(
                [np.array([x.get_constant() for x in c]) for c in self.segment_centers]
            )
            directions = np.array(
                [
                    np.array([d.get_constant() for d in c])
                    for c in self.segment_directions
                ]
            )
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
                    f"Coil type {type(self).__name__} does not have a well-defined axis for plotting."
                )

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
            Bfield: A Bfield object containing the field points and the
                    calculated Bvec objects.
        """
        # Input validation for field_points and backend
        if not isinstance(field_points, np.ndarray) or (
            field_points.ndim == 2 and field_points.shape[1] != 3
        ):
            raise TypeError("field_points must be a NumPy array of shape (N, 3).")
        if backend not in ["python", "cpp", "c", "cpp_v2", "mpi"]:
            raise ValueError(f"Backend '{backend}' is not a valid option.")

        # Check if segments have been generated
        if self.segment_centers is None:
            raise RuntimeError("Coil segments have not been generated.")

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
            b_field_vectors = np.array(
                [
                    Bvec(
                        self.current * vec[0],
                        self.current * vec[1],
                        self.current * vec[2],
                    )
                    for vec in b_field_vectors
                ],
                dtype=object,
            )
        else:
            b_field_vectors *= self.current

        # Apply post-processing based on use_mtf_for_segments flag for all coils
        for i, vec in enumerate(b_field_vectors):
            if self.use_mtf_for_segments:
                # When using MTF, a proper numerical integration is performed
                # over the segment, which is represented by a variable `u` in
                # the range [-1, 1].
                vec.Bx = vec.Bx.integrate(4, -1, 1)
                vec.By = vec.By.integrate(4, -1, 1)
                vec.Bz = vec.Bz.integrate(4, -1, 1)
            else:
                # When not using MTF, the field is calculated at a single point
                # (the center of the segment). The result is then multiplied by 2
                # as a numerical approximation to account for the contribution
                # of the segment's length.
                vec = 2 * vec
            b_field_vectors[i] = vec
        return Bfield(field_points=field_points, b_vectors=b_field_vectors)

    @staticmethod
    def plot_1d_field(
        coil_instance,
        field_component: str,
        axis: str = "x",
        start_point: np.ndarray = None,
        end_point: np.ndarray = None,
        num_points: int = 100,
        plot_type: str = "line",
        log_scale: bool = False,
        ax=None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **kwargs,
    ):
        """
        Plots a magnetic field component along a 1D line.
        """
        if field_component not in ["Bx", "By", "Bz", "B_norm"]:
            raise ValueError("field_component must be 'Bx', 'By', 'Bz', or 'B_norm'.")
        if (start_point is None and end_point is not None) or (
            start_point is not None and end_point is None
        ):
            raise ValueError(
                "start_point and end_point must both be provided or both be None."
            )

        if start_point is None:
            if axis not in ["x", "y", "z"]:
                raise ValueError(
                    "axis must be 'x', 'y', or 'z' if start_point is None."
                )

            # Auto-size the plot based on coil dimensions
            max_size = coil_instance.get_max_size()
            max_size = np.max(max_size).item()
            center = coil_instance.get_center_point()
            min_val = center[0] - 1.25 * max_size / 2
            max_val = center[0] + 1.25 * max_size / 2

            if axis == "x":
                line_points = np.linspace(min_val, max_val, num_points)
                field_points = np.vstack(
                    [
                        line_points,
                        np.full(num_points, center[1]),
                        np.full(num_points, center[2]),
                    ]
                ).T
                plot_axis_label = "x-axis"
            elif axis == "y":
                line_points = np.linspace(min_val, max_val, num_points)
                field_points = np.vstack(
                    [
                        np.full(num_points, center[0]),
                        line_points,
                        np.full(num_points, center[2]),
                    ]
                ).T
                plot_axis_label = "y-axis"
            elif axis == "z":
                line_points = np.linspace(min_val, max_val, num_points)
                field_points = np.vstack(
                    [
                        np.full(num_points, center[0]),
                        np.full(num_points, center[1]),
                        line_points,
                    ]
                ).T
                plot_axis_label = "z-axis"
        else:
            line_points = np.linspace(0, 1, num_points)
            field_points = np.array(
                [start_point + t * (end_point - start_point) for t in line_points]
            )
            plot_axis_label = "line"

        # Calculate the B-field
        bfield = coil_instance.biot_savart(field_points=field_points)

        # Extract the requested component
        if field_component == "Bx":
            field_values = np.array([b.Bx for b in bfield._b_vectors_mtf])
        elif field_component == "By":
            field_values = np.array([b.By for b in bfield._b_vectors_mtf])
        elif field_component == "Bz":
            field_values = np.array([b.Bz for b in bfield._b_vectors_mtf])
        elif field_component == "B_norm":
            field_values = bfield.B_norm

        # Evaluate the components if they are MTFs
        if isinstance(field_values[0], mtf):
            field_values = np.array([val.get_constant() for val in field_values])

        # Plot the data
        if ax is None:
            fig, ax = plt.subplots()

        if plot_type == "line":
            ax.plot(line_points, field_values, **kwargs)
        elif plot_type == "scatter":
            ax.scatter(line_points, field_values, **kwargs)
        else:
            raise ValueError("plot_type must be 'line' or 'scatter'.")

        # Customize plot
        if log_scale:
            ax.set_yscale("log")

        if not title:
            title = f"Field component {field_component} along {plot_axis_label}"
        if not xlabel:
            xlabel = plot_axis_label
        if not ylabel:
            ylabel = f"Magnetic field ({field_component})"

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if ax is None:
            plt.show()

    @staticmethod
    def plot_2d_field(
        coil_instance,
        field_component: str = "Bnorm",
        plane: str = "xy",
        center: np.ndarray = None,
        normal: np.ndarray = None,
        size_a: float = None,
        size_b: float = None,
        num_points_a: int = 50,
        num_points_b: int = 50,
        plot_type: str = "heatmap",
        ax=None,
        title: str = "",
        offset_from_center: float = 0.0,
        **kwargs,
    ):
        """
        Plots a magnetic field on a 2D plane.
        """
        if field_component not in ["Bx", "By", "Bz", "Bnorm"]:
            raise ValueError("field_component must be 'Bx', 'By', 'Bz', or 'Bnorm'.")
        if plot_type not in ["quiver", "streamline", "heatmap"]:
            raise ValueError("plot_type must be 'quiver', 'streamline', or 'heatmap'.")

        # Determine the plane and default center
        if center is None:
            center = coil_instance.get_center_point()

        # Automatically determine plot size if not specified
        if size_a is None or size_b is None:
            max_size = coil_instance.get_max_size()
            max_size = np.max(max_size).item()
            size_a = 1.25 * max_size if size_a is None else size_a
            size_b = 1.25 * max_size if size_b is None else size_b
            if plane == "xy":
                axis_labels = ("x", "y")
            elif plane == "yz":
                axis_labels = ("y", "z")
            elif plane == "xz":
                axis_labels = ("x", "z")
            else:
                if size_a is None or size_b is None:
                    raise ValueError(
                        "size_a and size_b must be specified for custom planes."
                    )

        # Grid generation
        if normal is None:
            if plane == "xy":
                a_coords = np.linspace(
                    center[0] - size_a / 2, center[0] + size_a / 2, num_points_a
                )
                b_coords = np.linspace(
                    center[1] - size_b / 2, center[1] + size_b / 2, num_points_b
                )
                A, B = np.meshgrid(a_coords, b_coords)
                C = np.full(A.shape, center[2])
                C = C + offset_from_center
                field_points = np.vstack([A.ravel(), B.ravel(), C.ravel()]).T

            elif plane == "yz":
                a_coords = np.linspace(
                    center[1] - size_a / 2, center[1] + size_a / 2, num_points_a
                )
                b_coords = np.linspace(
                    center[2] - size_b / 2, center[2] + size_b / 2, num_points_b
                )
                A, B = np.meshgrid(a_coords, b_coords)
                C = np.full(A.shape, center[0])
                C = C + offset_from_center
                field_points = np.vstack([C.ravel(), A.ravel(), B.ravel()]).T

            elif plane == "xz":
                a_coords = np.linspace(
                    center[0] - size_a / 2, center[0] + size_a / 2, num_points_a
                )
                b_coords = np.linspace(
                    center[2] - size_b / 2, center[2] + size_b / 2, num_points_b
                )
                A, B = np.meshgrid(a_coords, b_coords)
                C = np.full(A.shape, center[1])
                C = C + offset_from_center
                field_points = np.vstack([A.ravel(), C.ravel(), B.ravel()]).T

            else:
                raise ValueError(
                    "plane must be 'xy', 'yz', or 'xz' if normal is not provided."
                )
        else:
            # Generate grid for a custom plane
            normal = normal / np.linalg.norm(normal)
            if np.allclose(normal, np.array([0, 0, 1])) or np.allclose(
                normal, np.array([0, 0, -1])
            ):
                u = np.array([1, 0, 0])
            else:
                u = np.cross(normal, np.array([0, 0, 1]))
                u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            a_coords = np.linspace(-size_a / 2, size_a / 2, num_points_a)
            b_coords = np.linspace(-size_b / 2, size_b / 2, num_points_b)
            A, B = np.meshgrid(a_coords, b_coords)

            field_points = np.zeros((num_points_a * num_points_b, 3))
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    point = (
                        center + offset_from_center * normal + A[i, j] * u + B[i, j] * v
                    )
                    field_points[i * num_points_b + j] = point

        bfield = coil_instance.biot_savart(field_points=field_points)
        b_vectors = np.array([b.to_numpy_array() for b in bfield._b_vectors_mtf])

        if ax is None:
            fig, ax = plt.subplots()

        # Plotting logic
        if plot_type == "quiver":
            if normal is None:
                if plane == "xy":
                    U, V = b_vectors[:, 0], b_vectors[:, 1]
                elif plane == "yz":
                    U, V = b_vectors[:, 1], b_vectors[:, 2]
                elif plane == "xz":
                    U, V = b_vectors[:, 0], b_vectors[:, 2]
                ax.quiver(A, B, U.reshape(A.shape), V.reshape(B.shape), **kwargs)
            else:
                projected_b = (
                    b_vectors - np.dot(b_vectors, normal[:, np.newaxis]) * normal
                )
                U, V = np.dot(projected_b, u), np.dot(projected_b, v)
                ax.quiver(A, B, U.reshape(A.shape), V.reshape(B.shape), **kwargs)

        elif plot_type == "streamline":
            if normal is None:
                if plane == "xy":
                    U, V = b_vectors[:, 0], b_vectors[:, 1]
                elif plane == "yz":
                    U, V = b_vectors[:, 1], b_vectors[:, 2]
                elif plane == "xz":
                    U, V = b_vectors[:, 0], b_vectors[:, 2]
                ax.streamplot(A, B, U.reshape(A.shape), V.reshape(B.shape), **kwargs)
            else:
                projected_b = (
                    b_vectors - np.dot(b_vectors, normal[:, np.newaxis]) * normal
                )
                U, V = np.dot(projected_b, u), np.dot(projected_b, v)
                ax.streamplot(A, B, U.reshape(A.shape), V.reshape(B.shape), **kwargs)

        elif plot_type == "heatmap":
            if field_component == "Bnorm":
                field_data = bfield.get_magnitude()
            elif field_component == "Bx":
                field_data = np.array(
                    [b.Bx.get_constant() for b in bfield._b_vectors_mtf], dtype=float
                )
            elif field_component == "By":
                field_data = np.array(
                    [b.By.get_constant() for b in bfield._b_vectors_mtf], dtype=float
                )
            else:  # Bz
                field_data = np.array(
                    [b.Bz.get_constant() for b in bfield._b_vectors_mtf], dtype=float
                )
            field_data = np.real(field_data)
            c = ax.pcolormesh(A, B, field_data.reshape(A.shape), **kwargs)
            plt.colorbar(c, ax=ax)

        else:
            raise ValueError("plot_type must be 'quiver', 'streamline', or 'heatmap'.")

        # Set titles and labels
        if not title:
            title = f"Field {field_component} on {plane}-plane"
        ax.set_title(title)

        if normal is None:
            ax.set_xlabel(f"{axis_labels[0]}-axis")
            ax.set_ylabel(f"{axis_labels[1]}-axis")
        else:
            ax.set_xlabel("a-axis")
            ax.set_ylabel("b-axis")

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        if ax is None:
            plt.show()

    @staticmethod
    def plot_field_vectors_3d(
        coil_instance,
        num_points_a: int = 10,
        num_points_b: int = 10,
        num_points_c: int = 10,
        title: str = "",
        ax=None,
        **kwargs,
    ):
        """
        Generates a 3D quiver plot of the magnetic field vectors on a grid
        around the coil.

        This method automatically creates a grid of field points based on the
        coil's dimensions and then calculates and plots the magnetic field
        vectors at these points.

        Args:
            coil_instance (Coil): An instance of a Coil subclass.
            num_points_a (int): Number of grid points along the x-dimension.
            num_points_b (int): Number of grid points along the y-dimension.
            num_points_c (int): Number of grid points along the z-dimension.
            title (str, optional): The title of the plot. Defaults to an
                                   auto-generated title.
            ax (matplotlib.axes.Axes3D, optional): The 3D axis to plot on.
                                                  If None, a new figure is created.
            **kwargs: Additional keyword arguments for the `ax.quiver` function.
        """
        # Get the coil's bounding box to create a reasonable grid
        max_size = coil_instance.get_max_size()
        center = coil_instance.get_center_point()

        x_range = np.linspace(
            center[0] - 1.25 * max_size[0] / 2,
            center[0] + 1.25 * max_size[0] / 2,
            num_points_a,
        )
        y_range = np.linspace(
            center[1] - 1.25 * max_size[1] / 2,
            center[1] + 1.25 * max_size[1] / 2,
            num_points_b,
        )
        z_range = np.linspace(
            center[2] - 1.25 * max_size[2] / 2,
            center[2] + 1.25 * max_size[2] / 2,
            num_points_c,
        )

        # Create the grid of field points
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        field_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Calculate the magnetic field at these points
        bfield = coil_instance.biot_savart(field_points)
        b_vectors = np.array([b.to_numpy_array() for b in bfield._b_vectors_mtf])
        U, V, W = b_vectors[:, 0], b_vectors[:, 1], b_vectors[:, 2]

        # Reshape the 1D field component arrays to match the 3D meshgrid shape
        U = U.reshape(X.shape)
        V = V.reshape(Y.shape)
        W = W.reshape(Z.shape)

        # Get or create the 3D axes
        plot_ax = _get_3d_axes(ax)

        # Plot the vectors
        plot_ax.quiver(X, Y, Z, U, V, W, **kwargs)

        # Set title and labels
        if not title:
            title = f"3D Magnetic Field Vectors from {coil_instance.__class__.__name__}"
        plot_ax.set_title(title)
        plot_ax.set_xlabel("X-axis")
        plot_ax.set_ylabel("Y-axis")
        plot_ax.set_zlabel("Z-axis")

        plot_ax.set_aspect("equal", "box")

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
                radius, num_segments, center_point, axis_direction, use_mtf_for_segments
            )
        )

    def plot_1d_field(self, **kwargs):
        """Alias for Coil.plot_1d_field, automatically passing self."""
        Coil.plot_1d_field(self, **kwargs)

    def plot_2d_field(self, **kwargs):
        """Alias for Coil.plot_2d_field, automatically passing self."""
        Coil.plot_2d_field(self, **kwargs)

    def plot_field_vectors_3d(self, **kwargs):
        """Alias for Coil.plot_field_vectors_3d, automatically passing self."""
        Coil.plot_field_vectors_3d(self, **kwargs)

    @staticmethod
    def generate_geometry(
        ring_radius,
        num_segments_ring,
        ring_center_point,
        ring_axis_direction,
        use_mtf_for_segments=True,
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

        self.segment_centers, self.segment_lengths, self.segment_directions = (
            self.generate_geometry(
                p1, p2, p4, num_segments_per_side, use_mtf_for_segments
            )
        )

    def plot_1d_field(self, **kwargs):
        """Alias for Coil.plot_1d_field, automatically passing self."""
        Coil.plot_1d_field(self, **kwargs)

    def plot_2d_field(self, **kwargs):
        """Alias for Coil.plot_2d_field, automatically passing self."""
        Coil.plot_2d_field(self, **kwargs)

    def plot_field_vectors_3d(self, **kwargs):
        """Alias for Coil.plot_field_vectors_3d, automatically passing self."""
        Coil.plot_field_vectors_3d(self, **kwargs)

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
        self.segment_centers, self.segment_lengths, self.segment_directions = (
            self.generate_geometry(
                start_point, end_point, num_segments, use_mtf_for_segments
            )
        )

    def plot_1d_field(self, **kwargs):
        """Alias for Coil.plot_1d_field, automatically passing self."""
        Coil.plot_1d_field(self, **kwargs)

    def plot_2d_field(self, **kwargs):
        """Alias for Coil.plot_2d_field, automatically passing self."""
        Coil.plot_2d_field(self, **kwargs)

    def plot_field_vectors_3d(self, **kwargs):
        """Alias for Coil.plot_field_vectors_3d, automatically passing self."""
        Coil.plot_field_vectors_3d(self, **kwargs)

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
