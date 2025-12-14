"""
Vector and Vector Field representations.

This module provides classes for representing 3D vectors and vector fields,
with support for numerical data and `mtflib` Multivariate Taylor Functions (MTFs).
It includes classes like Vector, FieldVector, VectorField, and VectorFieldGrid.
"""

import matplotlib.pyplot as plt
import numpy as np

# Try to import mtflib. The code will still function with numerical data
# even if this import fails.
try:
    from mtflib import mtf

    _MTFLIB_AVAILABLE = True
except ImportError:
    _MTFLIB_AVAILABLE = False
    print("Warning: mtflib not found. Some functionality may be limited.")
    mtf = None  # type: ignore # To avoid NameError if used later

# # Placeholder imports for external libraries referenced in original code
# try:
#     from .biot_savart import serial_biot_savart, mpi_biot_savart, mpi_installed
# except ImportError:
#     pass  # Assume these will be provided externally if needed


class Vector:
    """
    A generic class to represent a 3D vector.
    This class handles standard vector operations like addition, subtraction,
    scalar multiplication/division, dot product, and cross product.
    """

    def __init__(self, *components):
        """
        Initializes the vector.

        This constructor is flexible and can accept arguments in several formats:
        - Three separate numeric or MTF values: `Vector(x, y, z)`
        - A list or tuple of three values: `Vector([x, y, z])`
        - A NumPy array of three values: `Vector(np.array([x, y, z]))`

        Args:
            *components (tuple): A tuple containing the components in one of the
                                  formats listed above.
        """
        if len(components) == 1 and isinstance(
            components[0], (list, tuple, np.ndarray)
        ):
            components = components[0]

        if len(components) != 3:
            raise ValueError(
                "Vector initialization requires 3 components, but "
                f"{len(components)} were given."
            )

        self.x, self.y, self.z = components

    @classmethod
    def from_array_of_vectors(cls, array):
        """
        Creates a NumPy array of Vector objects from a 2D NumPy array.

        Args:
            array (np.ndarray): A NumPy array of shape (N, 3), where N is the
                                number of vectors.

        Returns:
            np.ndarray: A NumPy array of Vector objects.
        """
        if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[1] != 3:
            raise TypeError("Input must be a 2D NumPy array with a shape of (N, 3).")
        return np.array([cls(row) for row in array])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Enables the use of NumPy universal functions (ufuncs) with Vector objects.
        This method is called when a NumPy ufunc is applied to a Vector instance.
        It allows for seamless operations with other NumPy arrays, scalars, and
        other Vector objects.
        """
        # Convert all Vector inputs to their NumPy array representation
        out_inputs = []
        for inp in inputs:
            if isinstance(inp, Vector):
                out_inputs.append(inp.to_numpy_array())
            elif (
                isinstance(inp, np.ndarray)
                and inp.dtype == object
                and all(isinstance(v, Vector) for v in inp.flat)
            ):
                # Convert a NumPy array of Vector objects to a 2D numerical array
                out_inputs.append(
                    np.array([v.to_numpy_array() for v in inp.flatten()]).reshape(
                        inp.shape + (3,)
                    )
                )
            else:
                out_inputs.append(inp)

        # Check if the method is supported
        if method == "__call__":
            # Apply the ufunc to the components
            result = ufunc(*out_inputs, **kwargs)

            # Handle the various possible result types
            if isinstance(result, np.ndarray):
                if result.ndim == 1 and result.shape[0] == 3:
                    return Vector(result)
                elif result.ndim > 1 and result.shape[-1] == 3:
                    # Return an array of Vector objects by reshaping the result
                    return np.array([
                        Vector(row) for row in result.reshape(-1, 3)
                    ]).reshape(result.shape[:-1])
                # If result is not a 3-element array or a (...,3) array,
                # return as is
                return result
            # If the result is not a NumPy array, return it as-is
            # (e.g., a scalar from a reduction)
            return result

        # Defer to NumPy's default behavior for other methods like 'reduce'
        return NotImplemented

    def __add__(self, other):
        """
        Adds another Vector object to this one.

        Args:
            other (Vector): The Vector object to add.

        Returns:
            Vector: A new Vector object representing the sum.
        """
        if isinstance(other, Vector):
            return type(self)(self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError(
            f"unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'"
        )

    def __sub__(self, other):
        """
        Subtracts another Vector object from this one.

        Args:
            other (Vector): The Vector object to subtract.

        Returns:
            Vector: A new Vector object representing the difference.
        """
        if isinstance(other, Vector):
            return type(self)(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError(
            f"unsupported operand type(s) for -: 'Vector' and '{type(other).__name__}'"
        )

    def __mul__(self, other):
        """
        Multiplies all components of the Vector by a scalar.

        Args:
            other (float or int): The scalar to multiply by.

        Returns:
            Vector: A new Vector object with scaled components.
        """
        if isinstance(other, (float, int)) or (
            _MTFLIB_AVAILABLE and isinstance(other, mtf)
        ):
            return type(self)(self.x * other, self.y * other, self.z * other)
        raise TypeError(
            f"unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'"
        )

    def __rmul__(self, other):
        """
        Handles right-hand side scalar multiplication (e.g., 5 * Vector).
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divides all components of the Vector by a non-zero scalar.

        Args:
            other (float or int): The scalar to divide by.

        Returns:
            Vector: A new Vector object with scaled components.
        """
        if not isinstance(other, (float, int)):
            raise TypeError(
                "unsupported operand type(s) for /: 'Vector' and "
                f"'{type(other).__name__}'"
            )
        if other == 0:
            raise ZeroDivisionError("cannot divide a Vector by zero")
        return self.__mul__(1.0 / other)

    def dot(self, other):
        """
        Calculates the dot product with another Vector object.

        Args:
            other (Vector): The Vector object to take the dot product with.

        Returns:
            mtf.MultivariateTaylorFunction or float: The scalar result of the
                dot product.
        """
        if not isinstance(other, Vector):
            raise TypeError(
                "unsupported operand type(s) for dot product: 'Vector' and "
                f"'{type(other).__name__}'"
            )

        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """
        Calculates the cross product with another Vector object.

        Args:
            other (Vector): The Vector object to take the cross product with.

        Returns:
            Vector: A new Vector object representing the resulting vector.
        """
        if not isinstance(other, Vector):
            raise TypeError(
                "unsupported operand type(s) for cross product: 'Vector' and "
                f"'{type(other).__name__}'"
            )

        x_component = self.y * other.z - self.z * other.y
        y_component = self.z * other.x - self.x * other.z
        z_component = self.x * other.y - self.y * other.x

        return type(self)(x_component, y_component, z_component)

    def norm(self):
        """
        Calculates the magnitude (L2-norm) of the vector.

        Returns:
            float or mtf.MultivariateTaylorFunction: The scalar magnitude.
        """
        squared_norm = self.dot(self)
        if _MTFLIB_AVAILABLE and isinstance(squared_norm, mtf):
            return mtf.sqrt(squared_norm)
        else:
            return np.sqrt(squared_norm)

    def is_mtf(self):
        """
        Checks if any of the Vector's components are MTF objects.

        Returns:
            bool: True if any component is an MTF, False otherwise.
        """
        if not _MTFLIB_AVAILABLE:
            return False
        return any(isinstance(comp, mtf) for comp in [self.x, self.y, self.z])

    def to_numpy_array(self):
        """
        Converts the vector to a NumPy array.

        This method now handles components that are either numbers or
        Multivariate Taylor Function (MTF) objects. For MTFs, it extracts the
        constant part of the function.
        """
        comps = []
        for comp in [self.x, self.y, self.z]:
            if _MTFLIB_AVAILABLE and isinstance(comp, mtf):
                # If it's an MTF, get its constant part
                comps.append(
                    comp.extract_coefficient(tuple([0] * comp.dimension)).item()
                )
            else:
                # Otherwise, assume it's a number
                comps.append(comp)
        return np.array(comps, dtype=np.complex128)

    def to_dataframe(self, column_names):
        """
        Converts the Vector components into a pandas DataFrame.

        This method is a helper for creating a clean, tabular representation
        of the vector, especially when components are MTF objects.

        Args:
            column_names (list of str): A list of three strings to be used
                                        as the column names for the components.

        Returns:
            pandas.DataFrame: A DataFrame representing the vector's components
                              and their coefficients if they are MTFs.
        """
        import pandas as pd

        if not self.is_mtf():
            data = {
                column_names[0]: [self.x],
                column_names[1]: [self.y],
                column_names[2]: [self.z],
            }
            return pd.DataFrame(data)

        # Handle MTF components
        dfs = {}
        for name, component in zip(column_names, [self.x, self.y, self.z]):
            if isinstance(component, mtf):
                df = component.get_tabular_dataframe()

                # Handle the case where the function is zero.
                if df.empty:
                    df = pd.DataFrame([
                        {
                            "Order": 0,
                            "Exponents": tuple([0] * component.dimension),
                            "Coefficient": 0.0,
                        }
                    ])

                df.rename(columns={"Coefficient": name}, inplace=True)
                df = df.sort_values(by=["Order", "Exponents"]).reset_index(drop=True)
                dfs[name] = df
            else:
                df = pd.DataFrame([
                    {"Order": 0, "Exponents": (0, 0, 0), name: component}
                ])
                dfs[name] = df

        # Merge the dataframes
        merged_df = pd.DataFrame()
        if column_names[0] in dfs:
            merged_df = dfs[column_names[0]]
        for name in column_names[1:]:
            if name in dfs:
                if merged_df.empty:
                    merged_df = dfs[name]
                else:
                    merged_df = pd.merge(
                        merged_df, dfs[name], on=["Order", "Exponents"], how="outer"
                    )

        # Fill NaN values with 0.0 for a cleaner table
        merged_df = merged_df.fillna(0.0)

        # Reorder columns to place 'Order' and 'Exponents' at the end
        cols = [col for col in merged_df.columns if col not in ["Order", "Exponents"]]
        reordered_cols = cols + ["Order", "Exponents"]
        merged_df = merged_df[reordered_cols]

        return merged_df

    def __str__(self):
        """
        Creates a basic string representation of the Vector object.
        """
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        """
        Provides a developer-friendly representation of the object.
        """
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"


class FieldVector(Vector):
    """
    Represents a vector at a point in a field, with components that can be
    Multivariate Taylor Functions (MTFs).
    """

    def __init__(self, x, y, z):
        """
        Initializes the field vector.

        Args:
            x (mtf.MultivariateTaylorFunction or float): The x-component.
            y (mtf.MultivariateTaylorFunction or float): The y-component.
            z (mtf.MultivariateTaylorFunction or float): The z-component.
        """
        super().__init__(x, y, z)

    def to_numpy_array(self):
        """
        Converts the Bvec to a NumPy array by extracting the constant part of
        each component.
        """
        comps = []
        for comp in [self.x, self.y, self.z]:
            if isinstance(comp, (int, float)):
                comps.append(comp)
            elif _MTFLIB_AVAILABLE and isinstance(comp, mtf):
                comps.append(
                    comp.extract_coefficient(tuple([0] * comp.dimension)).item()
                )
            else:
                raise TypeError(
                    "Components must be numerical or MTF objects to convert to a "
                    "NumPy array."
                )
        return np.array(comps, dtype=np.complex128)

    def curl(self):
        r"""
        Calculates the curl of the B-field vector.

        The curl of the B-field is given by the formula:
        $\nabla \times \mathbf{B} = (\frac{\partial B_z}{\partial y} -
        \frac{\partial B_y}{\partial z}) \mathbf{i} +
        (\frac{\partial B_x}{\partial z} - \frac{\partial B_z}{\partial x})
        \mathbf{j} + (\frac{\partial B_y}{\partial x} -
        \frac{\partial B_x}{\partial y}) \mathbf{k}$

        This method uses the `derivative` method from `mtflib` to compute the
        partial derivatives.

        Returns:
            FieldVector: A new FieldVector object representing the curl of the
                field.
        """
        # The variables of the MTF are assumed to be (x, y, z) corresponding to
        # dimensions 1, 2, 3
        curl_x = self.z.derivative(2) - self.y.derivative(3)
        curl_y = self.x.derivative(3) - self.z.derivative(1)
        curl_z = self.y.derivative(1) - self.x.derivative(2)

        return FieldVector(curl_x, curl_y, curl_z)

    def divergence(self):
        r"""
        Calculates the divergence of the B-field.

        The divergence of a vector field is a scalar value given by the formula:
        $\nabla \cdot \mathbf{B} = \frac{\partial B_x}{\partial x} +
        \frac{\partial B_y}{\partial y} + \frac{\partial B_z}{\partial z}$

        This method uses the `derivative` method from `mtflib` to compute the
        partial derivatives and then sums the resulting MTF objects.

        Returns:
            mtf.MultivariateTaylorFunction: A single MTF representing the
                                            scalar divergence of the field.
        """
        div_Bx = self.x.derivative(1)
        div_By = self.y.derivative(2)
        div_Bz = self.z.derivative(3)
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
        grad_Bx = np.array([
            self.x.derivative(1),
            self.x.derivative(2),
            self.x.derivative(3),
        ])
        grad_By = np.array([
            self.y.derivative(1),
            self.y.derivative(2),
            self.y.derivative(3),
        ])
        grad_Bz = np.array([
            self.z.derivative(1),
            self.z.derivative(2),
            self.z.derivative(3),
        ])

        return np.vstack([grad_Bx, grad_By, grad_Bz])

    def __str__(self):
        """
        Creates a string representation of the Bvec object.

        If components are MTFs, it returns a tabular format using the
        to_dataframe method. Otherwise, it uses the base class's string
        representation.
        """
        if not self.is_mtf():
            return super().__str__()

        import pandas as pd

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        df = self.to_dataframe(["x", "y", "z"])
        return df.to_string()

    def __repr__(self):
        """
        Provides a developer-friendly representation of the object.
        """
        return f"FieldVector(x={self.x}, y={self.y}, z={self.z})"


class VectorField:
    """
    A class to store a collection of FieldVector objects, representing a vector
    field at a set of discrete points in space.

    This class can handle both numerical and MTF-based field data.
    """

    def __init__(self, vectors, field_points=None):
        """
        Initializes the VectorField container.

        Args:
            vectors (np.ndarray): A NumPy array of FieldVector objects (if using MTF)
                                    or an (N, 3) NumPy array of vectors.
            field_points (np.ndarray, optional): A corresponding (N, 3)
                                                 NumPy array of numerical points or
                                                 an (N, 3) NumPy array of MTF objects.
                                                 Defaults to None.
        """
        if isinstance(vectors[0], FieldVector):
            # Case for MTF-based FieldVector objects
            if not isinstance(vectors, np.ndarray) or vectors.ndim != 1:
                raise TypeError(
                    "vectors must be a 1D NumPy array of FieldVector objects."
                )
            if not np.all(isinstance(v, FieldVector) for v in vectors):
                raise TypeError("All elements in vectors must be FieldVector objects.")

            self._vectors_mtf = vectors
            self._vectors_numerical = None
        else:
            # Case for numerical B-field vectors
            self._vectors_numerical = np.array(vectors)
            self._vectors_mtf = None

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
        if self._vectors_numerical is not None:
            # Data is already numerical
            if not isinstance(self.field_points, np.ndarray):
                raise TypeError(
                    "Numerical vector field data requires numerical field_points."
                )
            return self.field_points, self._vectors_numerical

        elif self._vectors_mtf is not None:
            if not _MTFLIB_AVAILABLE:
                raise RuntimeError(
                    "mtflib is required to evaluate FieldVector objects."
                )

            # Evaluate MTF FieldVectors to get numerical vectors
            vectors_numerical = np.array([
                [
                    v.x.extract_coefficient(tuple([0] * v.x.dimension)).item(),
                    v.y.extract_coefficient(tuple([0] * v.y.dimension)).item(),
                    v.z.extract_coefficient(tuple([0] * v.z.dimension)).item(),
                ]
                for v in self._vectors_mtf
            ])

            # Evaluate MTF field points to get numerical points
            if self.field_points is not None and self.field_points.size > 0:
                if isinstance(self.field_points[0][0], mtf):
                    numerical_points = np.array([
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
                    ])
                elif isinstance(self.field_points, np.ndarray):
                    numerical_points = self.field_points
                else:
                    raise TypeError("Unsupported type for field_points.")
            else:
                raise ValueError(
                    "VectorField object with MTF data must have corresponding "
                    "field_points."
                )

            return numerical_points, vectors_numerical
        else:
            raise ValueError("VectorField object does not contain any data.")

    def get_magnitude(self):
        """
        Calculates and returns the magnitude of each B-vector in the field.

        Returns:
            np.ndarray: A 1D NumPy array of the magnitudes.
        """
        if self._magnitude is None:
            if self._vectors_numerical is not None:
                self._magnitude = np.linalg.norm(self._vectors_numerical, axis=1)
            elif self._vectors_mtf is not None:
                if not _MTFLIB_AVAILABLE:
                    raise RuntimeError(
                        "mtflib is required to get magnitude of FieldVector objects."
                    )
                magnitudes = []
                for v in self._vectors_mtf:
                    norm = v.norm()
                    if v.is_mtf():
                        magnitudes.append(
                            norm.extract_coefficient(tuple([0] * v.x.dimension)).item()
                        )
                    else:
                        magnitudes.append(norm)
                self._magnitude = np.array(magnitudes)
        return self._magnitude

    def scatter(self, title="Vector Field Scatter Plot", ax=None, **kwargs):
        """
        Creates a 3D scatter plot of the vector field with direction arrows.
        The color and length of the arrows represent the magnitude of the vectors.
        This method provides a way to visualize the raw vector field, where
        arrow lengths are proportional to the vector strength.

        Args:
            title (str, optional): The title of the plot.
            ax (matplotlib.axes.Axes, optional): An existing 3D Axes
                                                 object to plot on.
            **kwargs: Additional keyword arguments passed to `ax.quiver`.
        """
        numerical_points, numerical_vectors = self._get_numerical_data()
        magnitudes = self.get_magnitude()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Ensure data is real for plotting
        numerical_points = numerical_points.real
        numerical_vectors = numerical_vectors.real
        magnitudes = magnitudes.real

        x = numerical_points[:, 0]
        y = numerical_points[:, 1]
        z = numerical_points[:, 2]
        u = numerical_vectors[:, 0]
        v = numerical_vectors[:, 1]
        w = numerical_vectors[:, 2]

        import matplotlib.cm as cm
        import matplotlib.colors as colors

        norm = colors.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
        cmap = kwargs.pop("cmap", plt.cm.viridis)

        rgba_colors = cmap(norm(magnitudes))

        ax.quiver(x, y, z, u, v, w, colors=rgba_colors, **kwargs)

        fig = plt.gcf()
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("B-field Magnitude")

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(title)

        if "show" not in kwargs or kwargs["show"]:
            plt.show()

    def quiver(self, dimension=3, title="Vector Field Quiver Plot", ax=None, **kwargs):
        """
        Creates a 3D quiver plot of the vector field.

        Args:
            title (str, optional): The title of the plot.
            ax (matplotlib.axes.Axes, optional): An existing 3D Axes
                                                 object to plot on.
            **kwargs: Additional keyword arguments passed to `ax.quiver`.
        """
        numerical_points, numerical_vectors = self._get_numerical_data()

        if dimension != 3:
            raise NotImplementedError("Only 3D quiver plots are currently supported.")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Ensure data is real for plotting
        numerical_points = numerical_points.real
        numerical_vectors = numerical_vectors.real

        x = numerical_points[:, 0]
        y = numerical_points[:, 1]
        z = numerical_points[:, 2]
        u = numerical_vectors[:, 0]
        v = numerical_vectors[:, 1]
        w = numerical_vectors[:, 2]

        # Default length=0.1 and normalize=True for better visualization
        length = kwargs.pop("length", 0.1)
        normalize = kwargs.pop("normalize", True)

        ax.quiver(x, y, z, u, v, w, length=length, normalize=normalize, **kwargs)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(title)

        if "show" not in kwargs or kwargs["show"]:
            plt.show()

    def max(self):
        """
        Returns the maximum magnitude of the B-field vectors.
        """
        magnitudes = self.get_magnitude()
        if isinstance(magnitudes[0], mtf):
            return max(m.get_constant() for m in magnitudes)
        return np.max(magnitudes)

    def min(self):
        """
        Returns the minimum magnitude of the B-field vectors.
        """
        magnitudes = self.get_magnitude()
        if isinstance(magnitudes[0], mtf):
            return min(m.get_constant() for m in magnitudes)
        return np.min(magnitudes)

    def to_dataframe(self):
        """
        Exports the field points and vector components to a pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame with columns
                              ['x', 'y', 'z', 'vx', 'vy', 'vz'].
        """
        import pandas as pd

        numerical_points, numerical_vectors = self._get_numerical_data()

        data = {
            "x": numerical_points[:, 0],
            "y": numerical_points[:, 1],
            "z": numerical_points[:, 2],
            "vx": numerical_vectors[:, 0],
            "vy": numerical_vectors[:, 1],
            "vz": numerical_vectors[:, 2],
        }

        return pd.DataFrame(data)


class VectorFieldGrid(VectorField):
    """
    A subclass of VectorField for data on a structured 1D, 2D, or 3D grid.

    This class is designed to hold vector field data that is arranged on a
    structured grid. It stores grid metadata (shape and axes) to allow for
    easy reshaping of the data for grid-based analysis and plotting.
    """

    def __init__(self, vectors, field_points, grid_shape, grid_axes):
        """
        Initializes the VectorFieldGrid container.

        Args:
            vectors (np.ndarray): A NumPy array of FieldVector objects or an
                                    (N, 3) NumPy array of vectors.
            field_points (np.ndarray): A corresponding (N, 3) NumPy array of
                                       numerical points or MTF objects.
            grid_shape (tuple): The shape of the grid (e.g., (50, 50) for a
                                2D grid).
            grid_axes (tuple): The axes of the grid (e.g., ('x', 'y')).
        """
        super().__init__(vectors, field_points)
        self.grid_shape = grid_shape
        self.grid_axes = grid_axes

        # Validate that the number of points matches the grid shape
        if np.prod(grid_shape) != len(self.field_points):
            raise ValueError(
                "The number of field points must match the product of the "
                "grid dimensions."
            )

    def get_grid_points(self):
        """
        Returns the field points reshaped into the grid dimensions.

        Returns:
            np.ndarray: The field points as a grid. The shape will be
                        ``(*grid_shape, 3)``.
        """
        numerical_points, _ = self._get_numerical_data()
        return numerical_points.reshape(*self.grid_shape, 3)

    def get_grid_vectors(self):
        """
        Returns the B-field vectors reshaped into the grid dimensions.

        Returns:
            np.ndarray: The B-field vectors as a grid. The shape will be
                        ``(*grid_shape, 3)``.
        """
        _, numerical_vectors = self._get_numerical_data()
        return numerical_vectors.reshape(*self.grid_shape, 3)


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

    # Initialize the VectorField object with the numerical data
    vector_field_num = VectorField(
        field_points=field_points_numerical, vectors=b_vectors_numerical
    )

    # Plot the 3D vector field using the new quiver method
    print("Plotting the 3D magnetic field vectors using quiver()...")
    vector_field_num.quiver(title="3D B-field Quiver Plot")

    # Plot the 3D vector field using the new scatter method
    print("Plotting the 3D B-field using scatter()...")
    vector_field_num.scatter(title="3D B-field Scatter Plot")

    # 2. Create dummy data with mtflib (if available)
    if _MTFLIB_AVAILABLE:
        print("\nCreating a VectorField object with a NumPy array of MTF objects...")
        mtf.initialize_mtf(max_order=2, max_dimension=3)

        # Create a grid of evaluation points using constant MTFs
        # Each point is a 3-element array of MTF objects
        field_points_mtf = np.array([
            [
                mtf.from_constant(p[0]),
                mtf.from_constant(p[1]),
                mtf.from_constant(p[2]),
            ]
            for p in field_points_numerical
        ])

        # Create a simple B-field as FieldVector objects.
        # This example assumes the B-field can be represented by a single
        # FieldVector object that is a function of the spatial variables,
        # and then we create an array of these objects for the VectorField
        # container.
        x_mtf, y_mtf, z_mtf = mtf.var(1), mtf.var(2), mtf.var(3)
        bvec_mtf_object = FieldVector(2 * x_mtf, 3 * y_mtf, 4 * z_mtf)
        b_vectors_mtf = np.array([bvec_mtf_object] * len(field_points_mtf))

        # Initialize the VectorField object with MTF objects
        vector_field_mtf = VectorField(
            field_points=field_points_mtf, vectors=b_vectors_mtf
        )

        print("Plotting the 3D magnetic field vectors from the MTF data...")
        vector_field_mtf.quiver(title="3D B-Field from MTF points")

    # 3. Test max, min, and to_dataframe
    print("\nTesting max, min, and to_dataframe methods...")
    print(f"Max B-field magnitude: {vector_field_num.max()}")
    print(f"Min B-field magnitude: {vector_field_num.min()}")
    df = vector_field_num.to_dataframe()
    print("B-field data as a pandas DataFrame:")
    print(df.head())

    # 4. Test VectorFieldGrid
    print("\nTesting VectorFieldGrid class...")
    grid_shape = (5, 5, 5)
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, grid_shape[0]),
        np.linspace(-1, 1, grid_shape[1]),
        np.linspace(-1, 1, grid_shape[2]),
    )
    field_points_grid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Create some dummy B-field vectors for the grid
    b_vectors_grid = np.random.rand(*field_points_grid.shape)

    vector_field_grid = VectorFieldGrid(
        vectors=b_vectors_grid,
        field_points=field_points_grid,
        grid_shape=grid_shape,
        grid_axes=("x", "y", "z"),
    )
    grid_points = vector_field_grid.get_grid_points()
    grid_vectors = vector_field_grid.get_grid_vectors()
    print(f"VectorFieldGrid grid_shape: {vector_field_grid.grid_shape}")
    print(f"Reshaped grid points shape: {grid_points.shape}")
    print(f"Reshaped grid vectors shape: {grid_vectors.shape}")
