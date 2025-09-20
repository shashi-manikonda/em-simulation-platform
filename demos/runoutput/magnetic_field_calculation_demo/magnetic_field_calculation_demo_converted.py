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
from em_app.currentcoils import RingCoil
from em_app import plotting
import matplotlib.pyplot as plt
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

# %% [markdown]
# ### 1a. Visualize the Geometry

# %%
ring_radius = 0.4
center_point = np.array([0, 0, 0])
axis_direction = np.array([0, 0, 1])
ring = RingCoil(1.0, ring_radius, 50, center_point, axis_direction)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ring.plot(ax)
ax.set_title("Geometry of the Current Ring")
plt.show()

# %% [markdown]
# ## 2. Calculate and Plot B-field

# %%
field_points = np.array([[0, 0, z] for z in np.linspace(-1, 1, 20)])
b_field_vectors = ring.biot_savart(field_points)
b_field_numerical = np.array([b.to_numpy_array() for b in b_field_vectors])

fig = plt.figure(figsize=(8, 6))
plt.plot(field_points[:, 2], b_field_numerical[:, 2], "b-o")
plt.title("Bz along the z-axis")
plt.xlabel("z-position")
plt.ylabel("Bz")
plt.grid(True)
plt.show()
