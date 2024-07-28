from typing import List

from wizard.base_game.card import Card

DELIMITER_COMBINED_REPRESENTATION = " - "


class ListCards:
    def __init__(self, cards: List[Card]):
        self.cards = cards

    def to_single_representation(
        self,
        delimiter: str = DELIMITER_COMBINED_REPRESENTATION,
        sort: bool = True,
    ):
        if sort:
            self.cards.sort(reverse=True)
        return delimiter.join(list(map(lambda card: card.representation, self.cards)))

    @classmethod
    def from_single_representation(
        cls,
        representation: str,
        delimiter: str = DELIMITER_COMBINED_REPRESENTATION,
        sort: bool = True,
    ):
        list_cards = cls(cards=list(map(Card.from_representation, representation.split(delimiter))))
        if sort:
            list_cards.cards.sort(reverse=True)
        return list_cards


def list_cards_to_representation_decorator(func):
    def inner(*args):
        return ListCards(func(*args)).to_single_representation()

    return inner
