from typing import List

import numpy as np

from config.common import (
    BASE_COLORS,
    JESTER_NAME,
    MAGICIAN_NAME,
    NUMBER_OF_JESTERS,
    NUMBER_OF_MAGICIANS,
    SUITS,
)
from wizard.base_game.card import Card


# noinspection PyTypeChecker
class Deck:
    def __init__(self, shuffle: bool = True):
        self.cards = self._create_new_deck(shuffle=shuffle)
        self.initial_cards = self.cards.copy()

    @staticmethod
    def _create_new_deck(shuffle: bool):
        list_cards = []
        for color in BASE_COLORS:
            for number in SUITS:
                list_cards += [Card(color=color, number=number, special_card=None)]
        list_cards += [
            Card(color=None, number=None, special_card=MAGICIAN_NAME)
        ] * NUMBER_OF_MAGICIANS
        list_cards += [
            Card(color=None, number=None, special_card=JESTER_NAME)
        ] * NUMBER_OF_JESTERS

        if shuffle:
            np.random.shuffle(list_cards)  # type: ignore

        return list_cards

    def filter_deck(
        self, colors: list[str], keep_joker: bool, keep_joker_duplicates: bool
    ):
        filtered_cards: list[Card] = []
        joker_already_added: list[str] = []
        for card in self.cards:
            if card.color in colors:
                filtered_cards += [card]
            elif (
                (card.special_card is not None)
                & keep_joker
                & (
                    keep_joker_duplicates
                    or card.special_card not in joker_already_added
                )
            ):
                filtered_cards += [card]
                joker_already_added += [card.special_card]
        return filtered_cards

    def shuffle(self) -> None:
        np.random.shuffle(self.cards)

    def remove_cards(self, cards_to_suppress: List[Card]):
        for card_to_suppress in cards_to_suppress:
            for card in self.cards:
                if card == card_to_suppress:
                    self.cards.remove(card)
                    break
        self.initial_cards = self.cards  # Useful for exhaustive_simulation purpose

    def reset_deck(self):
        self.cards = self.initial_cards.copy()
