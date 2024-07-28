from typing import Optional

from config.common import JESTER_NAME, MAGICIAN_NAME, TRUMP_COLOR
from wizard.base_game.card import Card
from wizard.base_game.player.player import Player


class PlayedCard:
    def __init__(
        self,
        card: Card,
        card_position: int,
        player: Player,
        starting_color: Optional[str] = None,
    ):
        self.card = card
        self.starting_color = starting_color
        self.card_position = card_position
        self.player = player

    def __gt__(self, other: object) -> bool:
        if isinstance(other, PlayedCard):
            if (self.card.special_card is None) & (other.card.special_card is None):
                if [
                    int(self.card.color == TRUMP_COLOR),
                    (int(self.card.color == self.starting_color)),
                    self.card.number,
                ] > [
                    int(other.card.color == TRUMP_COLOR),
                    (int(other.card.color == self.starting_color)),
                    other.card.number,
                ]:
                    return True
                return False
            if self.card_position < other.card_position:
                if self.card.special_card == MAGICIAN_NAME or other.card.special_card == JESTER_NAME:
                    return True
                return False
            if other.card.special_card == MAGICIAN_NAME or self.card.special_card == JESTER_NAME:
                return False
            return True
        return NotImplemented
