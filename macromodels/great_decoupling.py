from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

note = """
https://www.sciencedirect.com/science/article/pii/S0954349X25001596

Эта работа дает тебе вектор для развития твоего «детектора бреда». Если ты захочешь математически объяснить, почему на твоем графике Чехии в 2021 году произошел резкий излом и обвал реальных зарплат относительно производительности, эта статья подсказывает, куда смотреть. Тебе нужно будет проанализировать динамику корпоративного долга (финансиализация) и уровень инвестиций в промышленное оборудование (non-ICT) за этот же период.
"""


@dataclass(slots=True)
class GreatDecoupling[DT: pd.Series[int] | npt.NDArray]:
    """Great Decoupling model"""

    class ResultSchema(pa.DataFrameModel):
        labor_effectiveness: Series[np.number]
        real_wage: Series[np.number]

        value: Series[np.number]

    real_gdp: DT

    labor_time_worked: DT

    labor_cost: DT
    """D1_D4_MD5 should be used"""

    inflation: DT

    def __call__(self) -> DataFrame[ResultSchema]:
        df: DataFrame[GreatDecoupling.ResultSchema] = cast(
            DataFrame[GreatDecoupling.ResultSchema], pd.DataFrame()
        )
        # A = GDP / Hours
        df["labor_effectiveness"] = self.real_gdp / self.labor_time_worked * 100

        df["real_wage"] = self.labor_cost / self.inflation * 100

        df["value"] = df["labor_effectiveness"] - df["real_wage"]

        return df
