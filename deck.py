import numpy as np

from card import Card

NUMBER_OF_JESTER = 2
NUMBER_OF_MAGICIANS = 2
COLORS = ['RED', 'BLUE', 'GREEN', 'YELLOW']
SUITS = list(range(1, 14))


# noinspection PyTypeChecker
class Deck:
    def __init__(self, deck_id: int):
        self.cards = self._create_new_deck()
        self.deck_id = deck_id  # to be improved
        self.initial_cards = self.cards.copy()

    @staticmethod
    def _create_new_deck():
        list_cards = []
        for color in COLORS:
            for number in SUITS:
                list_cards += [Card(color=color, number=number, special_card=None)]
        list_cards += [Card(color=None, number=None, special_card='Wizard')] * NUMBER_OF_MAGICIANS
        list_cards += [Card(color=None, number=None, special_card='Jester')] * NUMBER_OF_JESTER

        np.random.shuffle(list_cards)

        return list_cards

    def reset_deck(self):
        self.cards = self.initial_cards

    def reset_deck_randomly(self):
        self.cards = self.initial_cards
        np.random.shuffle(self.cards)

    def filter_deck(self,
                    colors: list[str],
                    add_joker: bool,
                    keep_joker_duplicates: bool):
        filtered_cards: list[Card] = []
        joker_already_added: list[str] = []
        for card in self.cards:
            if card.color in colors:
                filtered_cards += [card]
            elif (card.special_card is not None) & add_joker & (
                    keep_joker_duplicates or card.special_card not in joker_already_added):
                filtered_cards += [card]
                joker_already_added += [card.special_card]
        return filtered_cards

    @property
    def number_cards_per_color(self):
        return len(SUITS)

    @property
    def number_color(self):
        return len(COLORS)

    @property
    def get_colors(self):
        return COLORS

    @property
    def number_jester(self):
        return NUMBER_OF_JESTER

    @property
    def number_wizard(self):
        return NUMBER_OF_MAGICIANS


class FilteredDeck(Deck):
    def __init__(self, id: int, cards_to_suppress):
        self.cards = self.create_filtered_new_deck(cards_to_suppress)
        self.id = id
        self.initial_cards = self.cards

    def create_filtered_new_deck(self, cards_to_suppress):
        cards = self._create_new_deck()
        for card_to_suppress in cards_to_suppress:
            for card in cards:
                if card.is_equal(card_to_suppress):
                    cards.remove(card)
                    break
        return cards

