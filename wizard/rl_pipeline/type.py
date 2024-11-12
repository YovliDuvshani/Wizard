from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    Prediction: "Prediction"
    CardToPlay: "CardToPlay"


@dataclass
class Action:
    type: ActionType
    value: int | str