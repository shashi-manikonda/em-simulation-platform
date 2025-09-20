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
# # Rectangular Loop Magnetic Field Demo

# %%
import numpy as np
from em_app.currentcoils import RectangularCoil, _rotation_matrix
from em_app import plotting
import matplotlib.pyplot as plt
from mtflib import mtf

mtf.initialize_mtf(max_order=6, max_dimension=4)

# %% [markdown]
# ## 1. Define Geometry and Visualize

# %%
width = 1.0
height = 1.0
position = np.array([0.5, 0.5, 0.5])
orientation_axis = np.array([0, 1, 1])
orientation_angle = np.pi / 4

p1_local = np.array([-width / 2, -height / 2, 0])
p2_local = np.array([width / 2, -height / 2, 0])
p4_local = np.array([-width / 2, height / 2, 0])

rotation = _rotation_matrix(orientation_axis, orientation_angle)
p1 = rotation @ p1_local + position
p2 = rotation @ p2_local + position
p4 = rotation @ p4_local + position

rect_loop = RectangularCoil(1.0, p1, p2, p4, 20)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
rect_loop.plot(ax)
ax.set_title("Geometry of the Rectangular Loop")
plt.show()

# %% [markdown]
# ## 2. Numerical B-Field Calculation

# %%
observation_point = np.array([[0.5, 0.5, -1.0]])
B_vector = rect_loop.biot_savart(observation_point)[0]
B_numerical = B_vector.to_numpy_array()

print(f"Computed B-field at the observation point: {B_numerical}")
