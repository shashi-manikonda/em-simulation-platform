import numpy as np
import pytest
from em_app.solvers import Backend, serial_biot_savart
from em_app.sources import StraightWire
from sandalwood import MultivariateTaylorFunction, mtf

# Check backend availability
try:
    from sandalwood.backends.cosy import cosy_backend

    COSY_AVAILABLE = cosy_backend.COSY_AVAILABLE
except ImportError:
    COSY_AVAILABLE = False


@pytest.mark.skipif(
    not COSY_AVAILABLE, reason="COSY backend required for optimization tests"
)
class TestCosyOptimization:
    def setup_method(self):
        # Reset global state to ensure clean initialization for each test
        MultivariateTaylorFunction._INITIALIZED = False
        MultivariateTaylorFunction.initialize_mtf(
            max_order=2, max_dimension=10, implementation="cosy"
        )

    def test_from_cosy_indices_factory(self):
        """
        Verify that MultivariateTaylorFunction.from_cosy_indices correctly wraps
        raw COSY indices into functional MTF objects.
        """
        dim = 10
        da1 = cosy_backend.CosyDA.from_const(1.0)
        da2 = cosy_backend.CosyDA.from_const(2.0)

        # Create copies
        da1_idx = (da1 + 0.0).idx
        da2_idx = (da2 + 0.0).idx

        indices = np.array([da1_idx, da2_idx], dtype=np.int32)
        mtfs = MultivariateTaylorFunction.from_cosy_indices(indices, dimension=dim)

        assert len(mtfs) == 2
        assert np.isclose(mtfs[0].get_constant(), 1.0)
        assert np.isclose(mtfs[1].get_constant(), 2.0)

        res = mtfs[0] + mtfs[1]
        assert np.isclose(res.get_constant(), 3.0)

    def test_biot_savart_fast_path_vs_parametric(self):
        """
        Verify that serial_biot_savart logic correctly dispatches and returns
        consistent results.
        """
        # Setup
        field_points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        # 1. Discrete Source (Fast Path)
        centers_f = np.array([[0.0, 0.0, 0.0]])
        lengths_f = np.array([0.1])
        dirs_f = np.array([[0.0, 0.0, 1.0]])

        bx_fast, by_fast, bz_fast = serial_biot_savart(
            centers_f, lengths_f, dirs_f, field_points, backend=Backend.COSY
        )
        assert bx_fast.dtype == np.float64
        assert by_fast.dtype == np.float64
        assert bz_fast.dtype == np.float64

        # 2. Parametric Source (MTF Path)
        # Create trivial MTFs (constants) to mimic floats
        centers_m = np.array(
            [[mtf.from_constant(0.0), mtf.from_constant(0.0), mtf.from_constant(0.0)]],
            dtype=object,
        )

        bx_param, by_param, bz_param = serial_biot_savart(
            centers_m, lengths_f, dirs_f, field_points, backend=Backend.COSY
        )
        assert bx_param.dtype == object

        # Value check
        # Check y-component at point 1 (index 0 in array? No, point 1 is index 0.
        # Wait test used index 0)
        # field_points has 2 points.
        # old code: res_fast[0, 1] means point 0, By component.

        val_fast = by_fast[0]
        val_param = by_param[0].get_constant()

        print(f"DEBUG: Fast[0,y] = {val_fast}")
        print(f"DEBUG: Param[0,y] = {val_param}")
        print(f"DEBUG: Diff = {abs(val_fast - val_param)}")

        assert np.isclose(val_fast, val_param, atol=1e-15)

    def test_dynamic_integration_variable(self):
        """
        Verify sources can use custom integration variable indices.
        """
        u_idx = 6

        wire = StraightWire(
            current=1.0,
            start_point=[0, 0, 0],
            end_point=[0, 0, 1],
            num_segments=1,
            integration_var_index=u_idx,
        )

        centers, _, _ = wire.get_segments()
        center_mtf = centers[0]

        z_comp = center_mtf.z

        # Check dependency on var(6)
        # Use dimension from component
        target_exp = [0] * z_comp.dimension
        if u_idx <= z_comp.dimension:
            target_exp[u_idx - 1] = 1
            coeff = z_comp.extract_coefficient(tuple(target_exp))
            assert abs(coeff) > 1e-10, f"Expected linear dependency on var({u_idx})"
        else:
            pytest.fail(f"Integration var index {u_idx} > dimension {z_comp.dimension}")
