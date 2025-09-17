import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .currentcoils import CurrentLoop

def _plot_loop_geometry(ax, loops, refine_level=1, color='blue', label=None):
    """
    Helper function to plot the geometry of the loops.
    """
    for loop in loops:
        segments = loop.get_segments()
        positions = segments[:, 0:3]
        dl_vectors = segments[:, 3:6]

        # Plot the segments
        for i in range(len(positions)):
            start_point = positions[i] - dl_vectors[i] / 2
            end_point = positions[i] + dl_vectors[i] / 2
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                label=label if i == 0 else ""
            )

def plot_field_on_line(loops, start_point, end_point, component='magnitude', num_points=100):
    """
    Plots a component of the B-field along a straight line and shows the loop geometry.
    """
    fig = plt.figure(figsize=(15, 7))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_2d = fig.add_subplot(122)

    _plot_loop_geometry(ax_3d, loops)

    line_points = np.linspace(start_point, end_point, num_points)
    ax_3d.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'r--', label='Observation Line')

    B_vectors = np.zeros((num_points, 3))
    for i, point in enumerate(line_points):
        B_total_at_point = np.zeros(3)
        for loop in loops:
            B_total_at_point += loop.biot_savart(np.array([point]))[0]
        B_vectors[i] = B_total_at_point

    distances = np.linalg.norm(line_points - start_point, axis=1)

    if component == 'magnitude':
        field_values = np.linalg.norm(B_vectors, axis=1)
    elif component == 'Bx':
        field_values = B_vectors[:, 0]
    elif component == 'By':
        field_values = B_vectors[:, 1]
    elif component == 'Bz':
        field_values = B_vectors[:, 2]
    else:
        raise ValueError("component must be one of 'magnitude', 'Bx', 'By', 'Bz'")

    ax_2d.plot(distances, field_values, 'k-')
    ax_2d.set_xlabel('Distance along line')
    ax_2d.set_ylabel(f'B-Field Component: {component}')
    ax_2d.set_title('Field Profile along Line')
    ax_2d.grid(True)

    ax_3d.legend()
    ax_3d.set_title('Loop Geometry and Observation Line')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.axis('equal')

    plt.tight_layout()
    return fig, (ax_3d, ax_2d)

def plot_field_on_plane(loops, center_point, normal_vector, size=(2, 2), resolution=(20, 20), plot_type='quiver', component='magnitude'):
    """
    Visualizes the B-field on a specified plane.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    _plot_loop_geometry(ax, loops)

    normal = np.array(normal_vector) / np.linalg.norm(normal_vector)
    u_vec = np.cross([0, 0, 1], normal)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)

    u_coords = np.linspace(-size[0] / 2, size[0] / 2, resolution[0])
    v_coords = np.linspace(-size[1] / 2, size[1] / 2, resolution[1])
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    points_3d = np.array(center_point) + u_grid[..., np.newaxis] * u_vec + v_grid[..., np.newaxis] * v_vec
    points_flat = points_3d.reshape(-1, 3)

    B_total = np.zeros_like(points_flat)
    for loop in loops:
        B_total += loop.biot_savart(points_flat)

    B_vectors_grid = B_total.reshape(points_3d.shape)
    B_magnitude = np.linalg.norm(B_total, axis=1).reshape(u_grid.shape)

    if plot_type == 'quiver':
        ax.plot_surface(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], alpha=0.2, facecolors=plt.cm.viridis(B_magnitude / B_magnitude.max()))
        ax.quiver(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2],
                  B_vectors_grid[..., 0], B_vectors_grid[..., 1], B_vectors_grid[..., 2],
                  length=np.mean(size) * 0.1, normalize=True, color='k')
    elif plot_type == 'contour':
        plot_data = B_magnitude
        ax.plot_surface(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], facecolors=plt.cm.viridis(plot_data / plot_data.max()), rstride=1, cstride=1, shade=False)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_data.min(), vmax=plot_data.max()), cmap='viridis'), ax=ax, shrink=0.5)

    ax.set_title('Field Visualization on a Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')

    return fig, ax
