import numpy as np

from tsfield import BoundaryMode, FieldConfiguration, FieldRuntime


def test_neumann_laplacian_constant_is_zero():
    rt = FieldRuntime(FieldConfiguration(resolution=5, boundary=BoundaryMode.NEUMANN))
    field = np.ones((5, 5, 5))
    assert np.allclose(rt.laplacian(field), 0)


def test_periodic_mode_is_explicit_and_receipted():
    rt = FieldRuntime(FieldConfiguration(resolution=5, boundary=BoundaryMode.PERIODIC))
    receipt = rt.step()
    assert receipt["boundary"] == "PERIODIC"
    assert rt.step_counter == 1
