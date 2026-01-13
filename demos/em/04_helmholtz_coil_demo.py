import matplotlib.pyplot as plt
import numpy as np
from em_app.solvers import calculate_b_field
from em_app.sources import RingCoil
from sandalwood import mtf

# Initialize the MTF library
# The 4th dimension is used for numerical integration along the segments.
mtf.initialize_mtf(max_order=1, max_dimension=4)

# --- 1. Setup the Helmholtz Coil Geometry ---
# A Helmholtz coil consists of two identical circular coils placed
# symmetrically along a common axis, separated by a distance equal to the
# radius of the coils.
radius = 0.5  # meters
separation = radius
current = 1.0  # Amperes
num_segments = 20  # Reduced for performance

# Create the first coil
coil1 = RingCoil(
    current=current,
    radius=radius,
    num_segments=num_segments,
    center_point=np.array([0, 0, -separation / 2]),
    axis_direction=np.array([0, 0, 1]),
)

# Create the second coil
coil2 = RingCoil(
    current=current,
    radius=radius,
    num_segments=num_segments,
    center_point=np.array([0, 0, separation / 2]),
    axis_direction=np.array([0, 0, 1]),
)

# --- 2. Define the Field Points for Calculation ---
# We will calculate the field on a 2D grid (XZ plane) to visualize it.
grid_size = 1.5 * radius
num_points = 20  # Reduced for performance
x_points = np.linspace(-grid_size, grid_size, num_points)
z_points = np.linspace(-grid_size, grid_size, num_points)
X, Z = np.meshgrid(x_points, z_points)
field_points = np.vstack([X.ravel(), np.zeros_like(X.ravel()), Z.ravel()]).T

# --- 3. Calculate the Magnetic Field ---
# Calculate the B-field from each coil and add them together.
print("Calculating magnetic field from Coil 1...")
b_field1 = calculate_b_field(coil1, field_points)
print("Calculating magnetic field from Coil 2...")
b_field2 = calculate_b_field(coil2, field_points)

# The total field is the vector sum of the fields from each coil
# Uses the new VectorField addition (SoA compatible)
total_b_field = b_field1 + b_field2

# Extract numerical vectors for plotting
_, b_vectors = total_b_field._get_numerical_data()

# --- 4. Plot the Results ---
print("Generating plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the coil geometry
coil1.plot(ax, color="b", wire_thickness=0.02)
coil2.plot(ax, color="r", wire_thickness=0.02)

# Plot the magnetic field vectors using quiver
ax.quiver(
    field_points[:, 0],
    field_points[:, 1],
    field_points[:, 2],
    b_vectors[:, 0],
    b_vectors[:, 1],
    b_vectors[:, 2],
    length=grid_size * 0.1,  # Scale arrow length
    normalize=True,
    color="gray",
)

# --- 5. Customize and Save the Plot ---
ax.set_title("Magnetic Field of a Helmholtz Coil")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_xlim([-grid_size, grid_size])
ax.set_ylim([-grid_size, grid_size])
ax.set_zlim([-grid_size, grid_size])
ax.set_aspect("equal", "box")
ax.view_init(elev=20.0, azim=-60)
plt.grid(True)

# Show the plot
plt.show()
print("Plot generated.")
