from dataclasses import dataclass

import pandas as pd
import pandera as pa
from pandera.typing.pandas import DataFrame, Series


class TaylorResultSchema(pa.DataFrameModel):

    taylor_rate: Series[float]
    actual_rate: Series[float]
    policy_spread: Series[float]
    """If policy spread:
    * greater than 0: Central Bank supresses economics
    * less than 0: CB has too soft campaign
    """


@dataclass(slots=True)
class TaylorRule:
    inflation: pd.Series
    output_gap: pd.Series
    actual_rate: pd.Series

    target_inflation: float | pd.Series = 2.0
    neutral_rate: float | pd.Series = 2.0

    inflation_weight: float = 0.5
    output_gap_weight: float = 0.5

    def __call__(self) -> DataFrame[TaylorResultSchema]:
        taylor_rate = (
            self.neutral_rate
            + self.inflation
            + self.inflation_weight * (self.inflation - self.target_inflation)
            + self.output_gap_weight * self.output_gap
        )

        policy_spread = self.actual_rate - taylor_rate

        df = pd.DataFrame(
            {
                "taylor_rate": taylor_rate,
                "actual_rate": self.actual_rate,
                "policy_spread": policy_spread,
            }
        )

        return df.dropna()
