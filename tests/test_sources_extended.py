import numpy as np
import pytest
from em_app.sources import StraightWire, RingCoil, Coil

def test_straight_wire_generation():
    """Verify StraightWire discretization and orientations."""
    start = np.array([0, 0, 0])
    end = np.array([0, 0, 2])
    n_seg = 5
    
    wire = StraightWire(current=1.0, start_point=start, end_point=end, num_segments=n_seg)
    
    centers, lengths, directions = wire.get_segments()
    
    assert len(centers) == n_seg
    assert np.allclose(lengths, 0.4)
    # Direction should be along Z
    for d in directions:
        assert np.allclose(d.to_numpy_array(), [0, 0, 1])

def test_coil_base_methods():
    """Test the methods provided by the base Coil class."""
    # We use a simple RingCoil to test the base methods
    coil = RingCoil(current=1.0, radius=1.0, num_segments=100, 
                    center_point=[0,0,0], axis_direction=[0,0,1])
    
    # Center point (arithmetic mean of segments)
    center = coil.get_center_point()
    assert np.allclose(center, [0, 0, 0], atol=1e-10)
    
    # Max size (bounding box)
    size = coil.get_max_size()
    # For a ring in XY plane with radius 1, size should be (2, 2, 0)
    # We use a coarser tolerance due to discretization overshoot
    assert np.allclose(size, [2.0, 2.0, 0.0], atol=1e-2)

def test_source_input_validation():
    """Ensure sources raise errors for invalid physical parameters."""
    with pytest.raises(ValueError, match="Radius must be a positive number."):
        RingCoil(current=1.0, radius=-1, num_segments=10, 
                 center_point=[0,0,0], axis_direction=[0,0,1])
                 
    with pytest.raises(ValueError, match="Start and end points cannot be the same"):
        StraightWire(current=1.0, start_point=[1,1,1], end_point=[1,1,1])
