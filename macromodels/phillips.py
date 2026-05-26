from dataclasses import dataclass
from typing import Annotated, NamedTuple

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


@dataclass(slots=True)
class PhillipsCurve:
    """New Keynesian Phillips Curve"""

    class Result(NamedTuple):
        alpha: np.float64
        beta: Annotated[np.float64, "Negative"]
        p_value: np.float64
        r_squared: np.float64

    inflation: npt.NDArray

    expected_inflation: npt.NDArray
    """expected_inflation - is inflation shifted by 1"""

    unemployment: npt.NDArray

    nairu: npt.NDArray

    # TODO: implement this in formula
    shocks: npt.NDArray | None = None

    def __post_init__(self) -> None:
        # validate dimensions
        if not (
            self.inflation.ndim
            == self.expected_inflation.ndim
            == self.unemployment.ndim
            == self.nairu.ndim
            == 1
        ):
            raise ValueError(
                f"{self.__class__.__name__} expects 1-dimensional array of data. Got: "
                f"indflation ({self.inflation.ndim}) "
                # TODO: fill
            )

        if not (
            len(self.inflation)
            == len(self.expected_inflation)
            == len(self.unemployment)
            == len(self.nairu)
        ):
            raise ValueError(
                f"{self.__class__.__name__} expects all data to be the same dtype."
            )

        if self.shocks is not None:
            if self.shocks.ndim != 1:
                raise ValueError(
                    f"{self.__class__.__name__} expects 1-dimensional array of shocks. Got: "
                )
            if len(self.shocks) != len(self.inflation):
                raise ValueError(
                    f"{self.__class__.__name__} expects all data to be the same dtype."
                )
        else:
            self.shocks = np.zeros(len(self.inflation), dtype=self.inflation.dtype)

    def __call__(self) -> PhillipsCurve.Result:
        delta_inflation: npt.NDArray[DT] = self.inflation - self.expected_inflation
        delta_unemployment: npt.NDArray[DT] = self.unemployment - self.nairu

        X = sm.add_constant(delta_unemployment)

        res = sm.OLS(delta_inflation, X, missing="drop").fit()

        return PhillipsCurve.Result(
            *res.params,
            res.pvalues[1],
            res.rsquared,
        )
