import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mtflib import MultivariateTaylorFunction
from .biot_savart import serial_biot_savart

class Coil:
    """
    A class to hold the information for a single coil or a collection of segments
    that form a single logical coil (e.g., a ring, a solenoid).
    """
    def __init__(self, segment_mtfs, element_lengths, direction_vectors, current=1.0, label=None, color='blue'):
        """
        Initializes a Coil object. Can accept MTF objects or NumPy arrays.
        """
        if isinstance(segment_mtfs, np.ndarray) and np.issubdtype(segment_mtfs.dtype, np.number):
            self.segment_mtfs = [[MultivariateTaylorFunction.from_constant(component) for component in vector] for vector in segment_mtfs]
        else:
            self.segment_mtfs = segment_mtfs

        if isinstance(direction_vectors, np.ndarray) and np.issubdtype(direction_vectors.dtype, np.number):
            self.direction_vectors = [[MultivariateTaylorFunction.from_constant(component) for component in vector] for vector in direction_vectors]
        else:
            self.direction_vectors = direction_vectors

        self.element_lengths = np.array(element_lengths)
        self.current = current
        self.label = label
        self.color = color

def _plot_coil_geometry(ax, coils, refine_level=1):
    """
    Helper function to plot the geometry of the coils.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The 3D axes object to plot on.
    coils : list of Coil
        A list of Coil objects to plot.
    refine_level : int, optional
        The number of points to evaluate along each segment for a smoother curve.
        A value of 1 will plot straight line segments. (default is 1)
    """
    for coil in coils:
        # The label should only be set for the first segment of the coil to avoid duplicate legend entries
        has_been_labeled = False
        for i, segment_mtf_components in enumerate(coil.segment_mtfs):
            # segment_mtf_components is an array of 3 MTF objects [mtf_x, mtf_y, mtf_z]
            u_vals = np.linspace(-1, 1, refine_level + 1)

            # Evaluate each component MTF for each u value
            # The 'u' variable in the MTF is the 4th dimension, so we create a point [0,0,0,u]
            points = np.array([[mtf.eval([0,0,0,u_val]) for mtf in segment_mtf_components] for u_val in u_vals])

            # Plot the refined segment
            label_to_use = coil.label if not has_been_labeled else ""
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=coil.color, label=label_to_use)
            if label_to_use:
                has_been_labeled = True

            # Add a quiver for current direction on the first segment of each coil
            if i == 0:
                # Evaluate at u=0 to get the midpoint of the segment
                mid_point_components = [mtf.eval([0,0,0,0]) for mtf in segment_mtf_components]
                mid_point = np.array([c[0] for c in mid_point_components])

                # Direction vector is also an MTF, evaluate it at u=0
                direction_components = [d.eval([0,0,0,0]) for d in coil.direction_vectors[i]]
                direction = np.array([c[0] for c in direction_components])

                ax.quiver(mid_point[0], mid_point[1], mid_point[2],
                          direction[0], direction[1], direction[2],
                          length=0.1 * np.mean(coil.element_lengths), normalize=True, color=coil.color, arrow_length_ratio=0.5)

def plot_field_on_line(coils, start_point, end_point, component='magnitude', num_points=100, refine_level=5, eval_order=0):
    """
    Plots a component of the B-field along a straight line and shows the coil geometry.

    Parameters:
    -----------
    coils : list of Coil
        A list of Coil objects that generate the magnetic field.
    start_point : array-like
        The [x, y, z] coordinates for the start of the line.
    end_point : array-like
        The [x, y, z] coordinates for the end of the line.
    component : str, optional
        The field component to plot. Options: 'magnitude', 'Bx', 'By', 'Bz'. (default is 'magnitude')
    num_points : int, optional
        The number of points to sample along the line. (default is 100)
    refine_level : int, optional
        Controls the smoothness of the plotted coil geometry. (default is 5)
    eval_order : int, optional
        The Taylor series order to use for field evaluation. (default is 0)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple of matplotlib.axes.Axes
        A tuple containing the (ax_3d, ax_2d) axes objects.
    """
    fig = plt.figure(figsize=(15, 7))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_2d = fig.add_subplot(122)

    # Plot coil geometry
    _plot_coil_geometry(ax_3d, coils, refine_level)

    # Plot the line on which the field is calculated
    line_points = np.linspace(start_point, end_point, num_points)
    ax_3d.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'r--', label='Observation Line')

    # Calculate the B-field at each point on the line
    B_vectors = np.zeros((num_points, 3), dtype=np.complex128)
    for i, point in enumerate(line_points):
        B_total_at_point = np.array([0.0j, 0.0j, 0.0j])
        for coil in coils:
            # Calculate the field contribution from this coil at the current point
            B_contrib_mtf = serial_biot_savart(
                coil.segment_mtfs,
                coil.element_lengths,
                coil.direction_vectors,
                np.array([point]), # Field point for calculation
                order=eval_order
            )
            # The result is a list containing one array of 3 MTFs
            # For zero-order, the constant term is the value
            B_vec_contrib = np.array([b.extract_coefficient(tuple([0]*b.dimension)).item() for b in B_contrib_mtf[0]])
            B_total_at_point += B_vec_contrib * coil.current

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
    ax_3d.set_title('Coil Geometry and Observation Line')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.axis('equal')

    plt.tight_layout()
    return fig, (ax_3d, ax_2d)

def plot_field_on_plane(coils, center_point, normal_vector, size=(2, 2), resolution=(20, 20), plot_type='quiver', component='magnitude', coords='rectangular', refine_level=5, eval_order=0):
    """
    Visualizes the B-field on a specified plane, embedded in a 3D view with the source coils.

    Parameters:
    -----------
    coils : list of Coil
        A list of Coil objects.
    center_point : array-like
        The [x, y, z] center of the plane.
    normal_vector : array-like
        The normal vector [nx, ny, nz] defining the plane's orientation.
    size : tuple, optional
        The (width, height) of the plotting area. (default is (2, 2))
    resolution : tuple, optional
        The grid resolution (e.g., (20, 20)). (default is (20, 20))
    plot_type : str, optional
        Type of plot: 'quiver', 'contour', 'streamplot'. (default is 'quiver')
    component : str, optional
        Scalar component for 'contour' plots. (default is 'magnitude')
    coords : str, optional
        Coordinate system on the plane: 'rectangular', 'polar'. (default is 'rectangular')
    refine_level : int, optional
        Smoothness of the plotted coil geometry. (default is 5)
    eval_order : int, optional
        Taylor series order for field evaluation. (default is 0)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The 3D axes object.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot coil geometry
    _plot_coil_geometry(ax, coils, refine_level)

    # Logic to create the grid on the plane
    # This involves creating a basis for the plane and generating points.
    # Placeholder for grid generation
    xx, yy = np.meshgrid(np.linspace(-size[0]/2, size[0]/2, resolution[0]),
                         np.linspace(-size[1]/2, size[1]/2, resolution[1]))

    # This involves creating a basis for the plane and generating points.

    # Normalize the normal vector
    normal = np.array(normal_vector) / np.linalg.norm(normal_vector)

    # Create an orthonormal basis for the plane
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u_vec = np.array([1, 0, 0])
    else:
        u_vec = np.cross([0, 0, 1], normal)
        u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)

    # Generate grid points in the plane's local coordinates
    if coords == 'rectangular':
        u_coords = np.linspace(-size[0] / 2, size[0] / 2, resolution[0])
        v_coords = np.linspace(-size[1] / 2, size[1] / 2, resolution[1])
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    elif coords == 'polar':
        r_coords = np.linspace(0, size[0] / 2, resolution[0])
        theta_coords = np.linspace(0, 2 * np.pi, resolution[1])
        r_grid, theta_grid = np.meshgrid(r_coords, theta_coords)
        u_grid = r_grid * np.cos(theta_grid)
        v_grid = r_grid * np.sin(theta_grid)
    else:
        raise ValueError("coords must be 'rectangular' or 'polar'")

    # Transform grid points to 3D space
    points_3d = np.array(center_point) + u_grid[..., np.newaxis] * u_vec + v_grid[..., np.newaxis] * v_vec
    points_flat = points_3d.reshape(-1, 3)

    # Calculate B-field at each grid point
    B_vectors = np.zeros_like(points_flat, dtype=np.complex128)
    for i, point in enumerate(points_flat):
        B_total_at_point = np.array([0.0j, 0.0j, 0.0j])
        for coil in coils:
            B_contrib_mtf = serial_biot_savart(
                coil.segment_mtfs, coil.element_lengths, coil.direction_vectors,
                np.array([point]), order=0
            )
            B_vec_contrib = np.array([b.extract_coefficient(tuple([0]*b.dimension)).item() for b in B_contrib_mtf[0]])
            B_total_at_point += B_vec_contrib * coil.current
        B_vectors[i] = B_total_at_point

    # Reshape B-vectors to match the grid shape
    B_vectors_grid = B_vectors.reshape(points_3d.shape)

    # Project B-field vectors onto the plane for 2D plotting
    B_u = np.dot(B_vectors, u_vec).reshape(u_grid.shape)
    B_v = np.dot(B_vectors, v_vec).reshape(v_grid.shape)
    B_normal = np.dot(B_vectors, normal).reshape(u_grid.shape)
    B_magnitude = np.linalg.norm(B_vectors, axis=1).reshape(u_grid.shape)

    # Plot the field on the plane
    if plot_type == 'quiver':
        # Plot the plane surface for context
        ax.plot_surface(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], alpha=0.2, facecolors=plt.cm.viridis(B_magnitude / B_magnitude.max()))
        # Plot quiver on the plane
        ax.quiver(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2],
                  B_vectors_grid[..., 0], B_vectors_grid[..., 1], B_vectors_grid[..., 2],
                  length=np.mean(size) * 0.1, normalize=True, color='k')
    elif plot_type == 'contour':
        if component == 'magnitude':
            plot_data = B_magnitude
        elif component == 'normal_component':
            plot_data = B_normal
        else:
            # Placeholder for Bx, By, Bz components if needed
            plot_data = B_magnitude

        c = ax.contourf(u_grid, v_grid, plot_data, cmap='viridis')
        # This requires transforming contour back to 3D, which is complex.
        # A simpler approach is to use plot_surface with colors.
        ax.plot_surface(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], facecolors=plt.cm.viridis(plot_data / plot_data.max()), rstride=1, cstride=1, shade=False)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_data.min(), vmax=plot_data.max()), cmap='viridis'), ax=ax, shrink=0.5)

    elif plot_type == 'streamplot':
        # Streamplot is inherently 2D, so we plot on a 2D axes and map the data
        fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
        ax_2d.streamplot(u_grid, v_grid, B_u, B_v, density=1.5, color=B_magnitude, cmap='viridis')
        ax_2d.set_title(f'Streamplot on Plane (Normal: {normal_vector})')
        ax_2d.set_xlabel('U-axis in plane')
        ax_2d.set_ylabel('V-axis in plane')
        ax_2d.set_aspect('equal')
        # This creates a separate 2D plot. Integrating it into the 3D plot is non-trivial.
        # For now, we will show it as a separate plot.

        # We can also plot the plane in the 3D plot for context
        ax.plot_surface(points_3d[..., 0], points_3d[..., 1], points_3d[..., 2], alpha=0.1)


    ax.set_title('Field Visualization on a Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')

    return fig, ax

def plot_field_vectors_3d(coils, points, refine_level=5, eval_order=0, scale=1.0, color_by_magnitude=True):
    """
    Generates a 3D plot showing field vectors at specified points in space,
    along with the source coil geometry.

    Parameters:
    -----------
    coils : list of Coil
        A list of Coil objects.
    points : Nx3 array-like
        An array of [x, y, z] coordinates where field vectors will be drawn.
    refine_level : int, optional
        Smoothness of the plotted coil geometry. (default is 5)
    eval_order : int, optional
        Taylor series order for field evaluation. (default is 0)
    scale : float, optional
        A factor to control the length of the plotted vectors. (default is 1.0)
    color_by_magnitude : bool, optional
        If True, the color of the vectors will correspond to the field's magnitude. (default is True)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The 3D axes object.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot coil geometry
    _plot_coil_geometry(ax, coils, refine_level)

    # Placeholder for B-field calculation
    # Calculate B-field at each specified point
    points = np.array(points)
    B_vectors = np.zeros_like(points, dtype=np.complex128)
    for i, point in enumerate(points):
        B_total_at_point = np.array([0.0j, 0.0j, 0.0j])
        for coil in coils:
            B_contrib_mtf = serial_biot_savart(
                coil.segment_mtfs, coil.element_lengths, coil.direction_vectors,
                np.array([point]), order=0
            )
            B_vec_contrib = np.array([b.extract_coefficient(tuple([0]*b.dimension)).item() for b in B_contrib_mtf[0]])
            B_total_at_point += B_vec_contrib * coil.current
        B_vectors[i] = B_total_at_point

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    U, V, W = B_vectors[:, 0], B_vectors[:, 1], B_vectors[:, 2]

    magnitudes = np.linalg.norm(B_vectors, axis=1)

    if color_by_magnitude and magnitudes.max() > 0:
        colors = plt.cm.viridis(magnitudes / magnitudes.max())
        q = ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=False, colors=colors, arrow_length_ratio=0.5)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max()), cmap='viridis'), ax=ax, shrink=0.5, label='B-Field Magnitude')
    else:
        q = ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=True, arrow_length_ratio=0.5)

    ax.set_title('3D Field Vector Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')

    return fig, ax
