from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    Prediction = "Prediction"
    CardToPlay = "CardToPlay"


@dataclass
class Action:
    type: ActionType
    value: int | str

    @property
    def is_prediction(self):
        return self.type == ActionType.Prediction

    @property
    def is_card_to_play(self):
        return self.type == ActionType.CardToPlay
