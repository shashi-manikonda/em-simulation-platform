"""
magneticfield: A library for magnetic field calculations and visualization.

This module defines classes and functions for working with magnetic fields,
with a focus on using Multivariate Taylor Functions (MTFs) from the mtflib
library.

The core components are:
- Bvec: A representation of a magnetic field vector at a single point,
        using MTFs for each component.
- Bfield: A container class for a collection of Bvec objects, providing
          methods for analysis and visualization, such as plotting the field
          on a plane or in 3D.
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt

# Try to import mtflib. The code will still function with numerical data
# even if this import fails.
try:
    from mtflib import mtf

    _MTFLIB_AVAILABLE = True
except ImportError:
    _MTFLIB_AVAILABLE = False
    print("Warning: mtflib not found. Some functionality may be limited.")
    mtf = None  # To avoid NameError if used later

# # Placeholder imports for external libraries referenced in original code
# try:
#     from .biot_savart import serial_biot_savart, mpi_biot_savart, mpi_installed
# except ImportError:
#     pass  # Assume these will be provided externally if needed


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
        grad_Bx = np.array(
            [self.Bx.derivative(1), self.Bx.derivative(2), self.Bx.derivative(3)]
        )
        grad_By = np.array(
            [self.By.derivative(1), self.By.derivative(2), self.By.derivative(3)]
        )
        grad_Bz = np.array(
            [self.Bz.derivative(1), self.Bz.derivative(2), self.Bz.derivative(3)]
        )

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

    This class can handle both numerical and MTF-based field data.
    """

    def __init__(self, b_vectors, field_points=None):
        """
        Initializes the Bfield container.

        Args:
            b_vectors (np.ndarray): A NumPy array of Bvec objects (if using MTF)
                                    or an (N, 3) NumPy array of B-field vectors.
            field_points (np.ndarray, optional): A corresponding (N, 3)
                                                 NumPy array of numerical points or
                                                 an (N, 3) NumPy array of MTF objects.
                                                 Defaults to None.
        """
        if isinstance(b_vectors[0], Bvec):
            # Case for MTF-based Bvec objects
            if not isinstance(b_vectors, np.ndarray) or b_vectors.ndim != 1:
                raise TypeError("b_vectors must be a 1D NumPy array of Bvec objects.")
            if not np.all(isinstance(v, Bvec) for v in b_vectors):
                raise TypeError("All elements in b_vectors must be Bvec objects.")

            self._b_vectors_mtf = b_vectors
            self._b_vectors_numerical = None
        else:
            # Case for numerical B-field vectors
            self._b_vectors_numerical = np.array(b_vectors)
            self._b_vectors_mtf = None

        if field_points is not None:
            field_points = np.array(field_points, dtype=object)
        self.field_points = field_points
        self._magnitude = None

    def _get_numerical_data(self):
        """
        Helper function to get numerical coordinates and vectors from the stored data,
        handling both NumPy arrays and MTF objects.

        Returns:
            tuple: A tuple containing (numerical_points, numerical_vectors).
        """
        if self._b_vectors_numerical is not None:
            # Data is already numerical
            if not isinstance(self.field_points, np.ndarray):
                raise TypeError(
                    "Numerical B-field data requires numerical field_points."
                )
            return self.field_points, self._b_vectors_numerical

        elif self._b_vectors_mtf is not None:
            if not _MTFLIB_AVAILABLE:
                raise RuntimeError("mtflib is required to evaluate Bvec objects.")

            # Evaluate MTF Bvecs to get numerical vectors
            b_vectors_numerical = np.array(
                [
                    [
                        v.Bx.extract_coefficient(tuple([0] * v.Bx.dimension)).item(),
                        v.By.extract_coefficient(tuple([0] * v.By.dimension)).item(),
                        v.Bz.extract_coefficient(tuple([0] * v.Bz.dimension)).item(),
                    ]
                    for v in self._b_vectors_mtf
                ]
            )

            # Evaluate MTF field points to get numerical points
            if self.field_points is not None and self.field_points.size > 0:
                if isinstance(self.field_points[0][0], mtf):
                    numerical_points = np.array(
                        [
                            [
                                p[0]
                                .extract_coefficient(tuple([0] * p[0].dimension))
                                .item(),
                                p[1]
                                .extract_coefficient(tuple([0] * p[1].dimension))
                                .item(),
                                p[2]
                                .extract_coefficient(tuple([0] * p[2].dimension))
                                .item(),
                            ]
                            for p in self.field_points
                        ]
                    )
                elif isinstance(self.field_points, np.ndarray):
                    numerical_points = self.field_points
                else:
                    raise TypeError("Unsupported type for field_points.")
            else:
                raise ValueError(
                    "Bfield object with MTF data must have corresponding field_points."
                )

            return numerical_points, b_vectors_numerical
        else:
            raise ValueError("Bfield object does not contain any data.")

    def get_magnitude(self):
        """
        Calculates and returns the magnitude of each B-vector in the field.

        Returns:
            np.ndarray: A 1D NumPy array of the magnitudes.
        """
        if self._magnitude is None:
            if self._b_vectors_numerical is not None:
                self._magnitude = np.linalg.norm(self._b_vectors_numerical, axis=1)
            elif self._b_vectors_mtf is not None:
                if not _MTFLIB_AVAILABLE:
                    raise RuntimeError(
                        "mtflib is required to get magnitude of Bvec objects."
                    )
                self._magnitude = np.array(
                    [
                        v.norm().extract_coefficient(tuple([0] * v.Bx.dimension)).item()
                        for v in self._b_vectors_mtf
                    ]
                )
        return self._magnitude

    def plot_field_on_plane(
        self, normal_vector, offset, title="B-Field Magnitude on Plane"
    ):
        """
        Plots the magnitude of the magnetic field on a 2D plane.

        This method projects the magnetic field data from the `Bfield` object
        onto the specified plane and visualizes its magnitude using a 2D color
        plot. It does not plot the source coils.

        Args:
            normal_vector (array-like): The normal vector [nx, ny, nz] defining
                                        the plane's orientation.
            offset (float): The scalar offset of the plane from the origin.
            title (str, optional): The title of the plot. Defaults to
                                   "B-Field Magnitude on Plane".
        """
        fig = plt.figure(figsize=(10, 8))
        fig.add_subplot(111, projection="3d")

        # Get numerical data
        points_3d, B_vec_components = self._get_numerical_data()

        # Normalize the normal vector
        normal = np.array(normal_vector) / np.linalg.norm(normal_vector)

        # Create an orthonormal basis for the plane
        if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
            u_vec = np.array([1, 0, 0])
        else:
            u_vec = np.cross([0, 0, 1], normal)
            u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(normal, u_vec)

        # Filter points that are on the plane
        on_plane_indices = np.where(
            np.isclose(np.dot(points_3d, normal), offset, atol=1e-6)
        )[0]

        if len(on_plane_indices) == 0:
            warnings.warn("No field points found on the specified plane.")
            plt.close(fig)
            return

        plane_points = points_3d[on_plane_indices]
        plane_b_vectors = B_vec_components[on_plane_indices]
        b_magnitudes = np.linalg.norm(plane_b_vectors, axis=1)

        # Project the points onto the 2D plane
        u_coords = np.dot(plane_points, u_vec)
        v_coords = np.dot(plane_points, v_vec)

        # Create the plot
        fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
        scatter = ax_2d.scatter(
            u_coords,
            v_coords,
            c=b_magnitudes,
            cmap="viridis",
            s=50,
            alpha=0.8,
        )
        cbar = fig_2d.colorbar(scatter)
        cbar.set_label("Magnetic Field Magnitude")

        ax_2d.set_title(title)
        ax_2d.set_xlabel("u (projected axis)")
        ax_2d.set_ylabel("v (projected axis)")
        ax_2d.set_aspect("equal", "box")
        ax_2d.grid(True)
        plt.show()

    def plot_field_vectors_3d(
        self, title="3D Magnetic Field Vectors", scale=1.0, color_by_magnitude=True
    ):
        """
        Generates a 3D quiver plot of the magnetic field vectors.

        Args:
            title (str, optional): The title of the plot. Defaults to
                                   "3D Magnetic Field Vectors".
            scale (float, optional): A factor to control the length of the
                                     plotted vectors. Defaults to 1.0.
            color_by_magnitude (bool, optional): If True, the color of the
                                                 vectors will correspond to the
                                                 field's magnitude.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Get numerical data
        points, B_vec_components = self._get_numerical_data()

        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        U, V, W = B_vec_components[:, 0], B_vec_components[:, 1], B_vec_components[:, 2]

        if color_by_magnitude:
            magnitudes = np.linalg.norm(B_vec_components, axis=1)
            if magnitudes.max() > 0:
                colors = plt.cm.viridis(magnitudes / magnitudes.max())
                ax.quiver(
                    X,
                    Y,
                    Z,
                    U,
                    V,
                    W,
                    length=scale,
                    normalize=True,
                    colors=colors,
                    arrow_length_ratio=0.5,
                )
                fig.colorbar(
                    plt.cm.ScalarMappable(
                        norm=plt.Normalize(
                            vmin=magnitudes.min(), vmax=magnitudes.max()
                        ),
                        cmap="viridis",
                    ),
                    ax=ax,
                    shrink=0.5,
                    label="B-Field Magnitude",
                )
            else:
                ax.quiver(
                    X,
                    Y,
                    Z,
                    U,
                    V,
                    W,
                    length=scale,
                    normalize=True,
                    arrow_length_ratio=0.5,
                )
        else:
            ax.quiver(
                X, Y, Z, U, V, W, length=scale, normalize=True, arrow_length_ratio=0.5
            )

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        plt.show()


if __name__ == "__main__":
    # --- Example Usage for Refactored Code ---
    # This block demonstrates how the new, refactored classes work.
    # In a real application, the field_points and B-vectors would be
    # generated by a magnetic field calculator (e.g., from `biot_savart`).
    print("Demonstrating the refactored Bfield class and plotting methods.")

    # 1. Create dummy numerical data
    print("\nCreating a Bfield object with numerical data...")
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)
    )
    field_points_numerical = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # A simple example B-field (e.g., from a z-axis dipole)
    r_cubed = (x**2 + y**2 + z**2) ** (3 / 2)
    b_vectors_x = 3 * x * z / r_cubed
    b_vectors_y = 3 * y * z / r_cubed
    b_vectors_z = (3 * z**2 - (x**2 + y**2 + z**2)) / r_cubed
    b_vectors_numerical = np.stack(
        [b_vectors_x.flatten(), b_vectors_y.flatten(), b_vectors_z.flatten()], axis=1
    )

    # Remove NaN values that can occur at the origin
    valid_indices = ~np.isnan(b_vectors_numerical).any(axis=1)
    field_points_numerical = field_points_numerical[valid_indices]
    b_vectors_numerical = b_vectors_numerical[valid_indices]

    # Initialize the Bfield object with the numerical data
    bfield_num = Bfield(
        field_points=field_points_numerical, b_vectors=b_vectors_numerical
    )

    # Plot the 3D vector field
    print("Plotting the 3D magnetic field vectors...")
    bfield_num.plot_field_vectors_3d()

    # Plot the field on the y=0 plane
    print("\nPlotting the magnetic field magnitude on the y=0 plane...")
    bfield_num.plot_field_on_plane(normal_vector=[0, 1, 0], offset=0.0)

    # 2. Create dummy data with mtflib (if available)
    if _MTFLIB_AVAILABLE:
        print("\nCreating a Bfield object with a NumPy array of MTF objects...")
        mtf.initialize_mtf(max_order=2, max_dimension=3)

        # Create a grid of evaluation points using constant MTFs
        # Each point is a 3-element array of MTF objects
        field_points_mtf = np.array(
            [
                [
                    mtf.from_constant(p[0]),
                    mtf.from_constant(p[1]),
                    mtf.from_constant(p[2]),
                ]
                for p in field_points_numerical
            ]
        )

        # Create a simple B-field as Bvec objects.
        # This example assumes the B-field can be represented by a single Bvec object
        # that is a function of the spatial variables, and then we create an array
        # of these objects for the Bfield container.
        x_mtf, y_mtf, z_mtf = mtf.var(1), mtf.var(2), mtf.var(3)
        bvec_mtf_object = Bvec(2 * x_mtf, 3 * y_mtf, 4 * z_mtf)
        b_vectors_mtf = np.array([bvec_mtf_object] * len(field_points_mtf))

        # Initialize the Bfield object with MTF objects
        bfield_mtf = Bfield(field_points=field_points_mtf, b_vectors=b_vectors_mtf)

        print("Plotting the 3D magnetic field vectors from the MTF data...")
        bfield_mtf.plot_field_vectors_3d(title="3D B-Field from MTF points")

        print(
            "\nPlotting the magnetic field magnitude from MTF data on the y=0 plane..."
        )
        bfield_mtf.plot_field_on_plane(
            normal_vector=[0, 1, 0], offset=0.0, title="B-Field from MTF on Plane"
        )
