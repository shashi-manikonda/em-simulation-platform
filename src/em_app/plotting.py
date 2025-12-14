"""
This module contains functions for plotting magnetic field data.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mtflib import mtf

from .solvers import calculate_b_field


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


def plot_1d_field(
    coil_instance,
    field_component: str,
    axis: str = "x",
    start_point: Optional[np.ndarray] = None,
    end_point: Optional[np.ndarray] = None,
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
    Plots a vector field component along a 1D line.
    """
    if field_component not in ["x", "y", "z", "norm"]:
        raise ValueError("field_component must be 'x', 'y', 'z', or 'norm'.")
    if (start_point is None and end_point is not None) or (
        start_point is not None and end_point is None
    ):
        raise ValueError(
            "start_point and end_point must both be provided or both be None."
        )

    if start_point is None:
        if axis not in ["x", "y", "z"]:
            raise ValueError("axis must be 'x', 'y', or 'z' if start_point is None.")

        # Auto-size the plot based on coil dimensions
        max_size = coil_instance.get_max_size()
        max_size = np.max(max_size).item()
        center = coil_instance.get_center_point()
        min_val = center[0] - 1.25 * max_size / 2
        max_val = center[0] + 1.25 * max_size / 2

        if axis == "x":
            line_points = np.linspace(min_val, max_val, num_points)
            field_points = np.vstack([
                line_points,
                np.full(num_points, center[1]),
                np.full(num_points, center[2]),
            ]).T
            plot_axis_label = "x-axis"
        elif axis == "y":
            line_points = np.linspace(min_val, max_val, num_points)
            field_points = np.vstack([
                np.full(num_points, center[0]),
                line_points,
                np.full(num_points, center[2]),
            ]).T
            plot_axis_label = "y-axis"
        elif axis == "z":
            line_points = np.linspace(min_val, max_val, num_points)
            field_points = np.vstack([
                np.full(num_points, center[0]),
                np.full(num_points, center[1]),
                line_points,
            ]).T
            plot_axis_label = "z-axis"
    else:
        line_points = np.linspace(0, 1, num_points)
        field_points = np.array([
            start_point + t * (end_point - start_point) for t in line_points
        ])
        plot_axis_label = "line"

    # Calculate the B-field
    vector_field = calculate_b_field(coil_instance, field_points=field_points)

    # Extract the requested component
    if field_component == "x":
        field_values = np.array([v.x for v in vector_field._vectors_mtf])
    elif field_component == "y":
        field_values = np.array([v.y for v in vector_field._vectors_mtf])
    elif field_component == "z":
        field_values = np.array([v.z for v in vector_field._vectors_mtf])
    elif field_component == "norm":
        field_values = vector_field.get_magnitude()

    # Evaluate the components if they are MTFs
    if isinstance(field_values[0], mtf):
        field_values = np.array([
            val.extract_coefficient(tuple([0] * val.dimension)).item()
            for val in field_values
        ])

    # Explicitly cast to float/real to avoid ComplexWarning
    field_values = np.real(field_values).astype(float)
    line_points = np.real(line_points).astype(float)

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
        ylabel = f"Vector field component ({field_component})"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if ax is None:
        plt.show()


def plot_2d_field(
    coil_instance,
    field_component: str = "norm",
    plane: str = "xy",
    center: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    size_a: Optional[float] = None,
    size_b: Optional[float] = None,
    num_points_a: int = 50,
    num_points_b: int = 50,
    plot_type: str = "heatmap",
    ax=None,
    title: str = "",
    offset_from_center: float = 0.0,
    **kwargs,
):
    """
    Plots a vector field on a 2D plane.
    """
    if field_component not in ["x", "y", "z", "norm"]:
        raise ValueError("field_component must be 'x', 'y', 'z', or 'norm'.")
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
            # Ensure coordinates are real
            A = np.real(A)
            B = np.real(B)
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
            # Ensure coordinates are real
            A = np.real(A)
            B = np.real(B)
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
            # Ensure coordinates are real
            A = np.real(A)
            B = np.real(B)
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
                point = center + offset_from_center * normal + A[i, j] * u + B[i, j] * v
                field_points[i * num_points_b + j] = point

    vector_field = calculate_b_field(coil_instance, field_points=field_points)
    b_vectors = np.array([b.to_numpy_array() for b in vector_field._vectors_mtf])

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
            # Explicitly cast to float to avoid ComplexWarning
            ax.quiver(
                A,
                B,
                np.real(U).reshape(A.shape).astype(float),
                np.real(V).reshape(B.shape).astype(float),
                **kwargs,
            )
        else:
            projected_b = b_vectors - np.dot(b_vectors, normal[:, np.newaxis]) * normal
            U, V = np.dot(projected_b, u), np.dot(projected_b, v)
            # Explicitly cast to float to avoid ComplexWarning
            ax.quiver(
                A,
                B,
                np.real(U).reshape(A.shape).astype(float),
                np.real(V).reshape(B.shape).astype(float),
                **kwargs,
            )

    elif plot_type == "streamline":
        if normal is None:
            if plane == "xy":
                U, V = b_vectors[:, 0], b_vectors[:, 1]
            elif plane == "yz":
                U, V = b_vectors[:, 1], b_vectors[:, 2]
            elif plane == "xz":
                U, V = b_vectors[:, 0], b_vectors[:, 2]
            # Explicitly cast to float to avoid ComplexWarning
            ax.streamplot(
                A,
                B,
                np.real(U).reshape(A.shape).astype(float),
                np.real(V).reshape(B.shape).astype(float),
                **kwargs,
            )
        else:
            projected_b = b_vectors - np.dot(b_vectors, normal[:, np.newaxis]) * normal
            U, V = np.dot(projected_b, u), np.dot(projected_b, v)
            # Explicitly cast to float to avoid ComplexWarning
            ax.streamplot(
                A,
                B,
                np.real(U).reshape(A.shape).astype(float),
                np.real(V).reshape(B.shape).astype(float),
                **kwargs,
            )

    elif plot_type == "heatmap":
        if field_component == "norm":
            field_data = vector_field.get_magnitude()
        elif field_component == "x":
            field_data = np.array([
                b.x.extract_coefficient(tuple([0] * b.x.dimension)).item()
                for b in vector_field._vectors_mtf
            ])
        elif field_component == "y":
            field_data = np.array([
                b.y.extract_coefficient(tuple([0] * b.y.dimension)).item()
                for b in vector_field._vectors_mtf
            ])
        else:  # z
            field_data = np.array([
                b.z.extract_coefficient(tuple([0] * b.z.dimension)).item()
                for b in vector_field._vectors_mtf
            ])
        # Explicitly cast to float/real
        field_data = np.real(field_data).astype(float)
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
        # Check for interactive backend to avoid warning in tests
        if plt.get_backend() != "Agg":
            plt.show()


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
    # Ensure coordinates are real
    X = np.real(X)
    Y = np.real(Y)
    Z = np.real(Z)
    field_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Calculate the magnetic field at these points
    vector_field = calculate_b_field(coil_instance, field_points)
    b_vectors = np.array([b.to_numpy_array() for b in vector_field._vectors_mtf])
    U, V, W = b_vectors[:, 0], b_vectors[:, 1], b_vectors[:, 2]

    # Reshape the 1D field component arrays to match the 3D meshgrid shape
    # Explicitly cast to float to avoid ComplexWarning
    U = np.real(U).reshape(X.shape).astype(float)
    V = np.real(V).reshape(Y.shape).astype(float)
    W = np.real(W).reshape(Z.shape).astype(float)

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
        # Check for interactive backend to avoid warning in tests
        if plt.get_backend() != "Agg":
            plt.show()
