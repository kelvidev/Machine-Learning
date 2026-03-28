import numpy as np
from dataclasses import dataclass
from typing import Self
import pandas as pd
@dataclass
class IrisFlower:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    iris_class: str

    @classmethod
    def from_json(cls, data: dict) -> Self:
        return cls(
            sepal_length=float(data["sepal_length"]),
            sepal_width=float(data["sepal_width"]),
            petal_length=float(data["petal_length"]),
            petal_width=float(data["petal_width"]),
            iris_class=data["class"],
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
            self.iris_class
        ]])
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
            self.iris_class,
        ]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])