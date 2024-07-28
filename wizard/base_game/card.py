from functools import cached_property
from typing import Optional

from config.common import BASE_COLORS, JESTER_NAME, MAGICIAN_NAME, SUITS


class Card:
    def __init__(
        self,
        color: Optional[str] = None,
        number: Optional[int] = None,
        special_card: Optional[str] = None,
    ):
        if special_card and (color or number):
            raise InvalidCard
        self.special_card = special_card
        self.number = number
        self.color = color

    @classmethod
    def from_representation(cls, card_representation: str):
        if card_representation in [JESTER_NAME, MAGICIAN_NAME]:
            return cls(special_card=card_representation)
        return cls(
            number=int(card_representation.split(" ")[0]),
            color=card_representation.split(" ")[1],
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            if ((self.number == other.number) & (self.color == other.color) & (self.special_card is None)) or (
                (self.special_card == other.special_card) & (self.special_card is not None)
            ):
                return True
            return False
        return NotImplemented

    def __gt__(self, other: object):
        if isinstance(other, Card):
            if (self.color is not None) and (other.color is not None):
                if [
                    -int(BASE_COLORS.index(self.color)),
                    self.number,
                ] > [
                    -int(BASE_COLORS.index(other.color)),
                    other.number,
                ]:
                    return True
                return False
            if self != other:
                if (self.special_card == MAGICIAN_NAME) or (other.special_card == JESTER_NAME):
                    return True
            return False
        return NotImplemented

    @property
    def representation(self) -> str:
        if self.color is not None:
            return f"{self.number} {self.color}"
        return str(self.special_card)

    @cached_property
    def id(self) -> int:
        if self.special_card == MAGICIAN_NAME:
            return 0
        elif self.special_card == JESTER_NAME:
            return 1
        return BASE_COLORS.index(self.color) * SUITS.index(self.number) + 2


class InvalidCard(Exception):
    pass
