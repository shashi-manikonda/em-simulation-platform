"""
Plotting Capabilities Demo
==========================

This example showcases the various field plotting capabilities of the `em_app`.
It demonstrates:

1.  **3D Geometry Plotting:** Visualizing the physical structure of current
    sources.
2.  **1D Field Plotting:** Plotting a specific component of the B-field along
    a line.
3.  **2D Field Plotting:** Visualizing the B-field as a heatmap and vector
    field on a plane.
4.  **3D Field Plotting:** Visualizing the B-field vectors in a 3D volume.

To do this, we first define a custom `HelmholtzCoil` class.
"""

import matplotlib.pyplot as plt
import numpy as np
from em_app.plotting import (
    plot_1d_field,
    plot_2d_field,
    plot_field_vectors_3d,
)
from em_app.sources import Coil, RingCoil
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)


# ## The Helmholtz Coil
#
# A Helmholtz coil is a special arrangement of two identical circular coils
# placed symmetrically along a common axis. When the current flows in the same
# direction in both coils, they produce a region of very uniform magnetic
# field in the center. Here, we define a `HelmholtzCoil` class that inherits
# from the base `Coil` class for easy use with our plotting functions.


class HelmholtzCoil(Coil):
    def __init__(self, current, radius, num_segments, center_point, axis_direction):
        super().__init__(current)
        self.coil1 = RingCoil(
            current,
            radius,
            num_segments,
            center_point - axis_direction * radius / 2,
            axis_direction,
        )
        self.coil2 = RingCoil(
            current,
            radius,
            num_segments,
            center_point + axis_direction * radius / 2,
            axis_direction,
        )
        self.segment_centers = np.concatenate(
            [
                self.coil1.segment_centers,
                self.coil2.segment_centers,
            ]
        )
        self.segment_lengths = np.concatenate(
            [
                self.coil1.segment_lengths,
                self.coil2.segment_lengths,
            ]
        )
        self.segment_directions = np.concatenate(
            [
                self.coil1.segment_directions,
                self.coil2.segment_directions,
            ]
        )
        self.current = current
        self.use_mtf_for_segments = self.coil1.use_mtf_for_segments

    def get_max_size(self):
        # A simple implementation for sizing the plots
        size1 = self.coil1.get_max_size()
        size2 = self.coil2.get_max_size()
        return np.maximum(size1, size2) * np.array([1, 1, 2])

    def get_center_point(self):
        return np.mean(
            [self.coil1.get_center_point(), self.coil2.get_center_point()], axis=0
        )

    def plot(self, ax=None, **kwargs):
        self.coil1.plot(ax, **kwargs)
        self.coil2.plot(ax, **kwargs)


# ## Part 1 & 2: Geometry Visualization
#
# First, we'll visualize the 3D geometry of a single `RingCoil` and our newly
# created `HelmholtzCoil`.

# --- Single Ring Coil Geometry ---
print("--- Visualizing Single Ring Coil ---")
ring_coil = RingCoil(1.0, 0.5, 10, np.array([0, 0, 0]), np.array([0, 0, 1]))

fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection="3d")
ring_coil.plot(ax=ax1)
ax1.set_title("Single Ring Coil Geometry")
plt.show()

# --- Helmholtz Coil Geometry ---
print("\n--- Visualizing Helmholtz Coil ---")
helmholtz_coil = HelmholtzCoil(1.0, 0.5, 10, np.array([0, 0, 0]), np.array([0, 0, 1]))

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection="3d")
helmholtz_coil.plot(ax=ax2)
ax2.set_title("Helmholtz Coil Geometry")
plt.show()

# ## Part 3: Field Plotting
#
# Now we'll use the `HelmholtzCoil` to demonstrate the different field
# plotting functions.

# ### 1D Field Plot
#
# Plot the z-component of the B-field along the central z-axis. Notice the
# flat, uniform region between the coils.

print("\n--- 1D Field Plot ---")
fig3, ax3 = plt.subplots()
plot_1d_field(helmholtz_coil, "z", axis="z", num_points=20, ax=ax3)
ax3.set_title("Z-component along the z-axis of a Helmholtz Coil")
plt.show()

# ### 2D Field Plot
#
# Plot the magnitude of the B-field as a heatmap on the XZ-plane, with
# vectors indicating the field direction.

print("\n--- 2D Field Plot ---")
fig4, ax4 = plt.subplots()
plot_2d_field(
    helmholtz_coil, "norm", plane="xz", num_points_a=5, num_points_b=5, ax=ax4
)
ax4.set_title("Vector Field Heatmap on the XZ-plane")
plt.show()

# ### 3D Field Plot
#
# Plot the B-field vectors in a 3D volume surrounding the Helmholtz coil.

print("\n--- 3D Field Plot ---")
fig5 = plt.figure(figsize=(10, 8))
ax5 = fig5.add_subplot(111, projection="3d")
plot_field_vectors_3d(
    helmholtz_coil, num_points_a=3, num_points_b=3, num_points_c=3, ax=ax5
)
helmholtz_coil.plot(ax=ax5, color="r")
ax5.set_title("3D B-field of a Helmholtz Coil")
plt.show()
