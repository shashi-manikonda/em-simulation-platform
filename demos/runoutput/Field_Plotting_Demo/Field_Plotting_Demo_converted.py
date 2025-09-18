#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # mtflib Demo: 3D Field Plotting Tools
#
# This notebook demonstrates the usage of the new 3D field plotting tools in `demos.applications.em.plotting`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from mtflib import *
from em_app.currentcoils import RingLoop, Pose
from em_app.plotting import (
    plot_field_on_line,
    plot_field_on_plane,
    plot_field_vectors_3d,
)
from mtflib import MultivariateTaylorFunction

MultivariateTaylorFunction.initialize_mtf(max_order=6, max_dimension=4)
MultivariateTaylorFunction.set_etol(1e-12)

# %% [markdown]
# ## 1. Define a Coil Configuration (Helmholtz Coil)
#
# We will create a Helmholtz coil, which consists of two identical circular coils placed symmetrically along a common axis.

# %%
radius = 0.5
separation = 0.5
current = 1.0
num_segments = 20

# First coil
center1 = np.array([0, 0, -separation / 2])
pose1 = Pose(
    position=center1, orientation_axis=np.array([1, 0, 0]), orientation_angle=0
)
coil1 = RingLoop(current=current, radius=radius, num_segments=num_segments, pose=pose1)

# Second coil
center2 = np.array([0, 0, separation / 2])
pose2 = Pose(
    position=center2, orientation_axis=np.array([1, 0, 0]), orientation_angle=0
)
coil2 = RingLoop(current=current, radius=radius, num_segments=num_segments, pose=pose2)

helmholtz_coils = [coil1, coil2]

# %% [markdown]
# ## 2. Demonstrate `plot_field_on_line`

# %%
start = [0.01, 0, -1]
end = [0.01, 0, 1]
fig, (ax3d, ax2d) = plot_field_on_line(helmholtz_coils, start, end, component="Bz")
plt.show()

# %% [markdown]
# ## 3. Demonstrate `plot_field_on_plane`

# %%
center = [0, 0.01, 0]
normal = [0, 1, 0]  # Plot on the x-z plane
fig, ax = plot_field_on_plane(
    helmholtz_coils,
    center,
    normal,
    plot_type="quiver",
    size=(2, 2),
    resolution=(15, 15),
)
plt.show()

# %% [markdown]
# ## 4. Demonstrate `plot_field_vectors_3d`

# %%
# Create a grid of points for the 3D vector plot
x_coords = np.linspace(-0.7, 0.7, 5)
y_coords = np.linspace(-0.7, 0.7, 5)
z_coords = np.linspace(-0.7, 0.7, 5)
points = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)

fig, ax = plot_field_vectors_3d(helmholtz_coils, points, scale=0.2)
plt.show()
