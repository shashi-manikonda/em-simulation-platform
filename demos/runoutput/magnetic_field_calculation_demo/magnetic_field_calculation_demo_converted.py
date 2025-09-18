# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # mtflib Demo: Magnetic Field Calculation and Validation
#
# This Jupyter Notebook demonstrates the calculation of the magnetic field of a current ring using `mtflib`.

# %% [markdown]
# ## 1. Import Libraries and Initialize

# %%
import numpy as np
from em_app import currentcoils, plotting
import matplotlib.pyplot as plt

# %% [markdown]
# ### 1a. Visualize the Geometry

# %%
ring_radius = 0.4
pose = currentcoils.Pose(
    position=np.array([0, 0, 0]),
    orientation_axis=np.array([0, 0, 1]),
    orientation_angle=0,
)
ring = currentcoils.RingLoop(
    current=1.0, radius=ring_radius, num_segments=50, pose=pose
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
plotting._plot_loop_geometry(ax, [ring])
ax.set_title("Geometry of the Current Ring")
plt.show()

# %% [markdown]
# ## 2. Calculate and Plot B-field

# %%
field_points = np.array([[0, 0, z] for z in np.linspace(-1, 1, 20)])
b_field = ring.biot_savart(field_points)

fig = plt.figure(figsize=(8, 6))
plt.plot(field_points[:, 2], b_field[:, 2], "b-o")
plt.title("Bz along the z-axis")
plt.xlabel("z-position")
plt.ylabel("Bz")
plt.grid(True)
plt.show()
