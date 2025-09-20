import pytest

def test_em_app_initialization():
    """
    Tests that the em_app package can be imported and that mtflib is
    initialized correctly.
    """
    try:
        import em_app
        from mtflib import mtf
        assert mtf._INITIALIZED
        assert mtf.get_max_order() == 5
        assert mtf.get_max_dimension() == 10
    except ImportError:
        pytest.fail("Failed to import em_app or mtflib")
    except AttributeError:
        pytest.fail("mtf._INITIALIZED does not exist. Need to find another way to check for initialization.")
