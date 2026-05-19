import numpy as np

from macromodels.phillips import PhillipsCurve


def test_phillips_curve():
    inflation = np.array([2.9, 2.3, 1.5, 2.2])
    expected_inflation = np.array([np.nan, 2.9, 2.3, 1.5])
    unemployment = np.array([5.4, 4.9, 4.5, 4.2])
    nairu = np.array([5.2, 5.2, 5.2, 5.2])

    pc = PhillipsCurve(inflation, expected_inflation, unemployment, nairu)
    result = pc()

    assert np.isclose(result.beta, -1.7297, atol=0.001)
    assert np.isclose(result.alpha, -1.3864, atol=0.001)
    assert np.isclose(result.p_value, 0.4640, atol=0.001)
    assert np.isclose(result.r_squared, 0.5562, atol=0.001)
