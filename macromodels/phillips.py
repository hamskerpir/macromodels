from dataclasses import dataclass
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(slots=True)
class PhillipsCurve:
    """New Keynesian Phillips Curve"""

    class Result(NamedTuple):
        alpha: np.float64
        beta: Annotated[np.float64, "Negative"]
        p_value: np.float64
        r_squared: np.float64

    inflation: pd.Series

    expected_inflation: pd.Series
    """expected_inflation - is inflation shifted by 1"""

    unemployment: pd.Series

    nairu: pd.Series

    # TODO: implement this in formula
    shocks: pd.Series | None = None

    def __call__(self) -> PhillipsCurve.Result:
        delta_inflation: pd.Series[np.number] = self.inflation - self.expected_inflation
        delta_unemployment: pd.Series[np.number] = self.unemployment - self.nairu

        X = sm.add_constant(delta_unemployment)

        res = sm.OLS(delta_inflation, X, missing="drop").fit()

        return PhillipsCurve.Result(
            *res.params,
            res.pvalues[1],
            res.rsquared,
        )
