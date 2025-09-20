# EM App

A Python package for calculating and visualizing magnetic fields from various current-carrying coils.

## Installation

You can install the package using pip:

```bash
pip install .
```

For development, install with the dev dependencies:
```bash
pip install -e .[dev]
```

## Quick Start

Here is a simple example of how to calculate the magnetic field of a circular coil:

```python
import numpy as np
from em_app import RingCoil, Bvec

# Define a circular coil
radius = 1.0
current = 1.0
num_segments = 50
center = np.array([0, 0, 0])
axis = np.array([0, 0, 1])

coil = RingCoil(current, radius, num_segments, center, axis)

# Define a field point
field_point = np.array([[0, 0, 1.0]])

# Calculate the magnetic field
b_vector = coil.biot_savart(field_point)[0]

print(f"Magnetic field at {field_point[0]}: {b_vector}")
```
