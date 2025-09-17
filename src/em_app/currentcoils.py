import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the mtflib and biot_savart libraries
from mtflib import mtf
from .biot_savart import serial_biot_savart, mpi_biot_savart, mpi_installed

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
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
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
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    v_cross = np.cross(v1_u, v2_u)
    if np.allclose(v_cross, 0):
        if np.dot(v1_u, v2_u) < 0:
            return _rotation_matrix(np.array([1, 0, 0]), np.pi)
        return np.eye(3)

    rotation_axis = v_cross / np.linalg.norm(v_cross)
    rotation_angle = np.arccos(
        np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    )
    return _rotation_matrix(rotation_axis, rotation_angle)

class Bvec:
    """
    Represents the magnetic field vector at a point as a set of
    Multivariate Taylor Functions (MTFs).
    """
    def __init__(self, Bx, By, Bz):
        """
        Initializes the B-field vector.

        Args:
            Bx (mtf.MultivariateTaylorFunction): The x-component of the B-field.
            By (mtf.MultivariateTaylorFunction): The y-component of the B-field.
            Bz (mtf.MultivariateTaylorFunction): The z-component of the B-field.
        """
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        
    def curl(self):
        """
        Calculates the curl of the B-field vector, which is a new B-field vector.
        
        The curl of the B-field is given by the formula:
        $\nabla \times \mathbf{B} = (\frac{\partial B_z}{\partial y} - \frac{\partial B_y}{\partial z}) \mathbf{i} + (\frac{\partial B_x}{\partial z} - \frac{\partial B_z}{\partial x}) \mathbf{j} + (\frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y}) \mathbf{k}$
        
        This method uses the `derivative` method from `mtflib` to compute the
        partial derivatives.

        Returns:
            Bvec: A new Bvec object representing the curl of the field.
        """
        # The variables of the MTF are assumed to be (x, y, z) corresponding to
        # dimensions 1, 2, 3
        # Curl B = (d(Bz)/dy - d(By)/dz)i + (d(Bx)/dz - d(Bz)/dx)j +
        # (d(By)/dx - d(Bx)/dy)k
        curl_x = self.Bz.derivative(2) - self.By.derivative(3)
        curl_y = self.Bx.derivative(3) - self.Bz.derivative(1)
        curl_z = self.By.derivative(1) - self.Bx.derivative(2)
        
        return Bvec(curl_x, curl_y, curl_z)
    
    def divergence(self):
        """
        Calculates the divergence of the B-field.
        
        The divergence of a vector field is a scalar value given by the formula:
        $\nabla \cdot \mathbf{B} = \frac{\partial B_x}{\partial x} + \frac{\partial B_y}{\partial y} + \frac{\partial B_z}{\partial z}$
        
        This method uses the `derivative` method from `mtflib` to compute the
        partial derivatives and then sums the resulting MTF objects.

        Returns:
            mtf.MultivariateTaylorFunction: A single MTF representing the
                                            scalar divergence of the field.
        """
        div_Bx = self.Bx.derivative(1)
        div_By = self.By.derivative(2)
        div_Bz = self.Bz.derivative(3)
        return div_Bx + div_By + div_Bz

    def gradient(self):
        """
        Calculates the Jacobian matrix of the B-field vector.
        
        The gradient of a vector field is a 3x3 matrix where each element
        is the partial derivative of a component of B with respect to a
        spatial variable.

        Returns:
            np.ndarray: A 3x3 array of MTFs representing the Jacobian matrix.
        """
        grad_Bx = np.array([self.Bx.derivative(1), self.Bx.derivative(2),
                            self.Bx.derivative(3)])
        grad_By = np.array([self.By.derivative(1), self.By.derivative(2),
                            self.By.derivative(3)])
        grad_Bz = np.array([self.Bz.derivative(1), self.Bz.derivative(2),
                            self.Bz.derivative(3)])
        
        return np.vstack([grad_Bx, grad_By, grad_Bz])

    def norm(self):
        """
        Calculates the magnitude (or L2 norm) of the B-field vector as a new MTF.
        
        The magnitude of the B-field is given by $|\mathbf{B}| = \sqrt{B_x^2 + B_y^2 + B_z^2}$.
        This method uses `mtf.sqrt` to compute the Taylor expansion of the
        magnitude.
        
        Returns:
            mtf.MultivariateTaylorFunction: A new MTF representing the magnitude
                                            of the magnetic field.
        """
        return mtf.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)

class Bfield:
    """
    A class to store a collection of Bvec objects, representing the magnetic
    field at a set of discrete points in space.
    """
    def __init__(self, b_vectors, field_points=None):
        """
        Initializes the Bfield container.

        Args:
            b_vectors (np.ndarray): A NumPy array of Bvec objects.
            field_points (np.ndarray, optional): A corresponding (N, 3)
                                                  NumPy array of the spatial
                                                  coordinates for each vector.
                                                  Defaults to None.
        """
        if not isinstance(b_vectors, np.ndarray) or b_vectors.ndim != 1:
            raise TypeError("b_vectors must be a 1D NumPy array.")
        if not np.all(isinstance(v, Bvec) for v in b_vectors):
            raise TypeError("All elements in b_vectors must be Bvec objects.")
            
        self.b_vectors = b_vectors
        self.field_points = field_points
        self._magnitude = None

    def get_magnitude(self):
        """
        Calculates and returns the magnitude of each Bvec in the field.

        Returns:
            np.ndarray: A 1D NumPy array of the magnitudes.
        """
        if self._magnitude is None:
            self._magnitude = np.array([
                v.norm().extract_coefficient(
                    tuple([0] * v.Bx.dimension)
                ).item() for v in self.b_vectors
            ])
        return self._magnitude

def _current_ring(ring_radius, num_segments_ring, ring_center_point,
                  ring_axis_direction, return_mtf=True):
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
    ring_axis_direction_unit = (ring_axis_direction /
                                np.linalg.norm(ring_axis_direction))

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

            center_point = np.array([x_center, y_center, z_center],
                                    dtype=object)
            center_point_rotated = np.dot(rotation_align_z_axis,
                                          center_point)
            center_point_translated = (center_point_rotated +
                                       ring_center_point_mtf)
            segment_mtfs_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            direction_base = np.array([-mtf.sin(phi), mtf.cos(phi),
                                       mtf.from_constant(0.0)],
                                      dtype=object)
            direction_rotated = np.dot(rotation_align_z_axis,
                                       direction_base)
            norm_mtf_squared = (direction_rotated[0] ** 2 +
                                direction_rotated[1] ** 2 +
                                direction_rotated[2] ** 2)
            norm_mtf_squared.set_coefficient((0, 0, 0, 0), 1.0)
            norm_mtf = mtf.sqrt(norm_mtf_squared)
            direction_normalized_mtf = [direction_rotated[i] / norm_mtf
                                        for i in range(3)]
            direction_vectors_ring.append(direction_normalized_mtf)

        return (np.array(segment_mtfs_ring, dtype=object),
                np.array(element_lengths_ring),
                np.array(direction_vectors_ring, dtype=object))

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
            center_point_rotated = (rotation_align_z_axis @
                                    np.array([x_center, y_center, z_center]))
            center_point_translated = center_point_rotated + ring_center_point
            segment_centers_ring.append(center_point_translated)

            element_lengths_ring.append(ring_radius * d_phi)

            # Tangent direction at center point (for base ring in xy-plane):
            direction_base = np.array([-np.sin(phi), np.cos(phi), 0])
            direction_rotated = rotation_align_z_axis @ direction_base
            direction_normalized = (direction_rotated /
                                    np.linalg.norm(direction_rotated))
            direction_vectors_ring.append(direction_normalized)

        return (np.array(segment_centers_ring),
                np.array(element_lengths_ring),
                np.array(direction_vectors_ring))

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
    wire_length_mtf = mtf.sqrt(mtf.sum(wire_vector_mtf ** 2))

    segment_length_mtf = wire_length_mtf / num_segments
    wire_direction_mtf = wire_vector_mtf / wire_length_mtf

    segment_centers = np.empty((num_segments, 3), dtype=object)
    
    for i in range(num_segments):
        factor = (i + 0.5) * segment_length_mtf
        segment_centers[i] = (start_point_mtf +
                              wire_direction_mtf * factor)

    segment_lengths = np.full(num_segments, segment_length_mtf)
    segment_directions = np.full((num_segments, 3), wire_direction_mtf,
                                 dtype=object)
    
    return segment_centers, segment_lengths, segment_directions


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
        all_coords = np.vstack([
            centers_numerical,
            centers_numerical + (directions_numerical *
                                 self.segment_lengths.reshape(-1, 1)),
            centers_numerical - (directions_numerical *
                                 self.segment_lengths.reshape(-1, 1))
        ])
        
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        
        return max_coords - min_coords

    def plot(self, ax=None, color='b'):
        """
        Plots the coil segments in a 3D matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes3D): The 3D axis to plot on.
            color (str): The color for the plotted segments.
        """
        if self.segment_centers is None:
            warnings.warn("No segments to plot.")
            return

        centers = mtf.to_numpy_array(self.segment_centers)
        directions = mtf.to_numpy_array(self.segment_directions)
        lengths = mtf.to_numpy_array(self.segment_lengths)

        for i in range(len(centers)):
            start_point = centers[i] - directions[i] * lengths[i] / 2
            end_point = centers[i] + directions[i] * lengths[i] / 2
            x_vals = [start_point[0], end_point[0]]
            y_vals = [start_point[1], end_point[1]]
            z_vals = [start_point[2], end_point[2]]
            if ax:
                ax.plot(x_vals, y_vals, z_vals, color=color)
            else:
                warnings.warn("No matplotlib axis provided for plotting.")

    def biot_savart(self, field_points, backend='python'):
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
        # Check if segments have been generated
        if self.segment_centers is None:
            raise RuntimeError("Coil segments have not been generated.")

        # The Biot-Savart formula is a sum over all segments.
        # B = (mu_0 / 4pi) * sum( (I * dl x r) / |r|^3 )
        # Here, we use the `serial_biot_savart` function to perform the calculation.

        # The function expects source points, dl vectors, and field points.
        source_points = self.segment_centers
        
        # Calculate dl vectors from lengths and directions
        dl_vectors = (self.segment_lengths[:, np.newaxis] *
                      self.segment_directions)
        
        if backend == 'mpi':
            if not mpi_installed:
                raise ImportError("The 'mpi' backend requires the 'mpi4py'"
                                  " library to be installed.")
            b_field_vectors = mpi_biot_savart(
                source_points=source_points,
                dl_vectors=dl_vectors,
                field_points=field_points
            )
        else:
            b_field_vectors = serial_biot_savart(
                source_points=source_points,
                dl_vectors=dl_vectors,
                field_points=field_points,
                backend=backend
            )
        
        b_field_vectors *= self.current # Scale by the current

        # Convert the numpy array of (Bx, By, Bz) to an array of Bvec objects
        if isinstance(b_field_vectors, np.ndarray):
            b_field_objects = []
            for vec in b_field_vectors:
                b_field_objects.append(Bvec(vec[0], vec[1], vec[2]))
            return np.array(b_field_objects, dtype=object)
        else:
            # Handle the case where the result is a single vector
            return np.array([Bvec(b_field_vectors[0], b_field_vectors[1],
                                    b_field_vectors[2])])


class RingCoil(Coil):
    """
    Represents a circular current-carrying coil.
    """
    def __init__(self, current, radius, num_segments, center_point,
                 axis_direction):
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
            _current_ring(radius, num_segments, center_point,
                          axis_direction, return_mtf=True)
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
                _straight_wire_segments(start_p, end_p,
                                        num_segments_per_side)
            )

        # Concatenate segments from all four sides
        self.segment_centers = np.concatenate(
            [s[0] for s in all_segments], axis=0
        )
        self.segment_lengths = np.concatenate(
            [s[1] for s in all_segments]
        )
        self.segment_directions = np.concatenate(
            [s[2] for s in all_segments], axis=0
        )

def _plot_coil_geometry(ax, coils, refine_level=5):
    """
    (PRIVATE) Plots the geometry of a list of coils.
    
    Args:
        ax (matplotlib.axes.Axes3D): The 3D axes object to plot on.
        coils (list of Coil): A list of Coil objects to plot.
        refine_level (int, optional): Level of refinement for plotting.
    """
    for coil in coils:
        coil.plot(ax)

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
    # Use Bfield and Coil classes for this
    B_vectors = []
    for coil in coils:
        B_vectors.append(coil.biot_savart(points_flat))
    
    # Sum the contributions from all coils
    B_total_vectors = np.sum(B_vectors, axis=0)

    # Convert the array of Bvec objects to a 3-column numpy array for plotting
    B_vec_components = np.array([
        [v.Bx.extract_coefficient(tuple([0]*v.Bx.dimension)).item(),
         v.By.extract_coefficient(tuple([0]*v.By.dimension)).item(),
         v.Bz.extract_coefficient(tuple([0]*v.Bz.dimension)).item()]
        for v in B_total_vectors
    ])
    
    # Reshape B-vectors to match the grid shape
    B_vectors_grid = B_vec_components.reshape(points_3d.shape)

    # Project B-field vectors onto the plane for 2D plotting
    B_u = np.dot(B_vec_components, u_vec).reshape(u_grid.shape)
    B_v = np.dot(B_vec_components, v_vec).reshape(v_grid.shape)
    B_normal = np.dot(B_vec_components, normal).reshape(u_grid.shape)
    B_magnitude = np.linalg.norm(B_vec_components, axis=1).reshape(u_grid.shape)

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

def plot_field_vectors_3d(B_total_vectors, points, refine_level=5, eval_order=0, scale=1.0, color_by_magnitude=True):
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
    # _plot_coil_geometry(ax, coils, refine_level)

    # # Calculate B-field at each specified point
    # points = np.array(points)
    
    # B_vectors = []
    # for coil in coils:
    #     B_vectors.append(coil.biot_savart(points))
    
    # # Sum the contributions from all coils
    # B_total_vectors = np.sum(B_vectors, axis=0)

    # Convert the array of Bvec objects to a 3-column numpy array for plotting
    B_vec_components = np.array([
        [v.Bx.extract_coefficient(tuple([0]*v.Bx.dimension)).item(),
         v.By.extract_coefficient(tuple([0]*v.By.dimension)).item(),
         v.Bz.extract_coefficient(tuple([0]*v.Bz.dimension)).item()]
        for v in B_total_vectors
    ])
    
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    U, V, W = B_vec_components[:, 0], B_vec_components[:, 1], B_vec_components[:, 2]

    magnitudes = np.linalg.norm(B_vec_components, axis=1)

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
