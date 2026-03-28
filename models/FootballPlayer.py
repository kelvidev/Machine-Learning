import numpy as np
from dataclasses import dataclass
from typing import Self
import pandas as pd

@dataclass
class FootballPlayer:
    player: str
    shots_on_goal: float
    disarms: float

    @classmethod
    def from_json(cls, data: dict) -> Self:
        return cls(
            player=data["player"],
            shots_on_goal=float(data["shots_on_goal"]),
            disarms=float(data["disarms"]),
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([[
            self.player,
            self.shots_on_goal,
            self.disarms,
        ]])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([[
            self.player,
            self.shots_on_goal,
            self.disarms,
        ]], columns=["player", "shots_on_goal", "disarms"])