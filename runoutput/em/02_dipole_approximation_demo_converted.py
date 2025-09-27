# %% [markdown]
# # Magnetic Dipole Approximation Validation
#
# This notebook demonstrates the validity of the magnetic dipole approximation for a current loop. It calculates the magnetic field from a `RingCoil` using the full Biot-Savart law and compares it to the field calculated using the dipole approximation at various distances.
#
# The key takeaway is that the dipole approximation becomes more accurate as the observation point moves farther away from the current loop.

# %%
import matplotlib.pyplot as plt
import numpy as np
from mtflib import mtf

from em_app.solvers import calculate_b_field
from em_app.sources import RingCoil

# %%
mtf.initialize_mtf(max_order=6, max_dimension=4)


# %% [markdown]
# ## The Magnetic Dipole B-Field
#
# At distances far from a current loop, the magnetic field can be approximated by that of a perfect magnetic dipole. The formula for the magnetic field of a dipole with magnetic moment **m** at a position **r** is:
#
# $$ \vec{B}_{dipole} = \frac{\mu_0}{4 \pi r^3} (3(\vec{m} \cdot \hat{r})\hat{r} - \vec{m}) $$
#
# where the magnetic moment **m** for a current loop is given by **m** = I * **A**, with **A** being the vector area of the loop.


# %%
def magnetic_dipole_b_field(magnetic_moment, r_vec):
    """
    Calculates the B-field from a magnetic dipole.
    """
    mu_0 = 4 * np.pi * 1e-7
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag

    term1 = 3 * np.dot(magnetic_moment, r_hat) * r_hat
    term2 = magnetic_moment

    b_field = (mu_0 / (4 * np.pi * r_mag**3)) * (term1 - term2)
    return b_field


# %% [markdown]
# ## Simulation Setup
#
# We will create a `RingCoil` and then calculate the relative error between the full Biot-Savart law calculation and the dipole approximation at increasing distances from the coil.

# %%
# --- Setup ---
current = 1.0
radius = 0.1
ring_coil = RingCoil(
    current,
    radius,
    num_segments=20,
    center_point=np.array([0, 0, 0]),
    axis_direction=np.array([0, 0, 1]),
)

# Magnetic moment of the ring coil
area = np.pi * radius**2
magnetic_moment = current * area * np.array([0, 0, 1])

# --- Calculation ---
distances = np.logspace(0, 3, 20) * radius  # from 1 to 1000 radii
errors = []

for d in distances:
    observation_point = np.array([[0, 0, d]])

    # Full Biot-Savart calculation
    b_field = calculate_b_field(ring_coil, observation_point)
    b_field_numerical = b_field._vectors_mtf[0].to_numpy_array()

    # Dipole approximation
    b_field_dipole = magnetic_dipole_b_field(magnetic_moment, observation_point[0])

    # Calculate relative error
    error = np.linalg.norm(b_field_numerical - b_field_dipole) / np.linalg.norm(
        b_field_numerical
    )
    errors.append(error)

# %% [markdown]
# ## Error Visualization
#
# Finally, we plot the relative error as a function of the distance from the coil, normalized by the coil's radius. The log-log plot clearly shows that the error decreases significantly as the distance increases.

# %%
plt.figure(figsize=(8, 6))
plt.loglog(distances / radius, errors, "b-o")
plt.title("Dipole Approximation Error vs. Distance")
plt.xlabel("Distance from coil (in radii)")
plt.ylabel("Relative Error")
plt.grid(True, which="both", ls="--")
plt.show()
