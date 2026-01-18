
EM-App User Guide & Reference
=============================

User Guide
----------

**1. Getting Started**

To simulate a magnetic field, define a coil and a set of points:

.. code-block:: python

    from em_app.sources import RingCoil
    from em_app.solvers import calculate_b_field
    import numpy as np

    # Define a coil
    coil = RingCoil(current=1.0, radius=0.5, center_point=[0,0,0], axis_direction=[0,0,1])

    # Define points (N, 3) array
    points = np.random.rand(100, 3)

    # Calculate B-field
    # Returns a VectorField object (SoA optimized)
    b_field = calculate_b_field(coil, points)

.. note::
    **Performance Optimization (SoA)**: As of v0.2.0, this function returns a ``VectorField`` using a **Structure of Arrays (SoA)** memory layout. 
    The components ``(Bx, By, Bz)`` are stored as contiguous numpy arrays rather than lists of vector objects. 
    This significantly improves memory locality and SIMD vectorization performance.

**2. High-Performance Backends**


EM-App uses `sandalwood` for accelerated computation. The solver automatically selects the best kernel ("Hybrid Dispatch"):

*   **Discrete Mode (Fast Path):** If you use standard numerical inputs (float arrays) and `use_mtf_for_segments=False` (default), the solver uses a raw Fortran kernel (~100x speedup). This is ideal for field mapping and particle tracking where derivatives are not needed.
*   **Parametric Mode (DA Mode):** If you pass `sandalwood.mtf` objects or enable `use_mtf_for_segments=True` in the Coil geometry, the solver uses Differential Algebra. This computes high-order derivatives and Taylor maps but is computationally more expensive.

**3. Memory Management**

When running large parametric simulations (DA mode), COSY requires explicit memory management to avoid exhausting the static Fortran memory pool. Use `CosyScope` to prevent stack overflows:

.. code-block:: python

    from sandalwood.backends.cosy_scope import CosyScope

    # Frees DA memory after block exits
    with CosyScope():
         b_field = calculate_b_field(coil, points)


Module Reference
----------------

em\_app.plotting module
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: em_app.plotting
   :members:
   :show-inheritance:
   :undoc-members:

em\_app.solvers module
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: em_app.solvers
   :members:
   :show-inheritance:
   :undoc-members:

em\_app.sources module
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: em_app.sources
   :members:
   :show-inheritance:
   :undoc-members:

em\_app.vector\_fields module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: em_app.vector_fields
   :members:
   :show-inheritance:
   :undoc-members:
