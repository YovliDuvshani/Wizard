from typing import Optional

import numpy as np

from config.common import (
    BASE_COLORS,
    SUITS,
    NUMBER_OF_MAGICIANS,
    JESTER_NAME,
    MAGICIAN_NAME,
    NUMBER_OF_JESTERS,
)
from wizard.card import Card


# noinspection PyTypeChecker
class Deck:
    def __init__(self, shuffle: Optional[bool] = True):
        self.cards = self._create_new_deck(shuffle=shuffle)

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
            np.random.shuffle(list_cards)

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

    @property
    def number_cards_per_color(self):
        return len(SUITS)

    @property
    def number_colors(self):
        return len(BASE_COLORS)

    @property
    def number_jesters(self):
        return NUMBER_OF_JESTERS

    @property
    def number_wizards(self):
        return NUMBER_OF_MAGICIANS


class FilteredDeck(Deck):
    def __init__(self, cards_to_suppress):
        self.cards = self.create_filtered_new_deck(cards_to_suppress)

    def create_filtered_new_deck(self, cards_to_suppress):
        cards = self._create_new_deck()
        for card_to_suppress in cards_to_suppress:
            for card in cards:
                if card == card_to_suppress:
                    cards.remove(card)
                    break
        return cards
