from enum import Enum
from typing import Optional

from config.common import BASE_COLORS

Color = Enum("COLOR", {color.lower(): color for color in BASE_COLORS})


class Card:
    def __init__(
        self,
        color: Optional[Color] = None,
        number: Optional[int] = None,
        special_card: Optional[str] = None,
    ):
        if special_card and (color or number):
            raise InvalidCard
        self.special_card = special_card
        self.number = number
        self.color = color

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            if ((self.number == other.number) & (self.color == other.color)) or (
                (self.special_card == other.special_card)
                & (self.special_card is not None)
            ):
                return True
            return False
        return NotImplemented

    @property
    def representation(self) -> str:
        if self.color is not None:
            return f"{self.number} {self.color}"
        return str(self.special_card)


class InvalidCard(Exception):
    pass
