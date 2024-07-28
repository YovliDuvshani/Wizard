from config.common import BASE_COLORS
from wizard.base_game.card import Card
from wizard.base_game.deck import Deck
from wizard.base_game.list_cards import ListCards
from wizard.exhaustive_simulation.hand_combinations import HandCombinationsTwoCards

hand_combination_class = HandCombinationsTwoCards(Deck())

all_possible_combinations = hand_combination_class.build_all_possible_hand_combinations()

list_cards = [
    Card(color=BASE_COLORS[2], number=10),
    Card(color=BASE_COLORS[3], number=10),
]
ListCards(hand_combination_class.list_cards_to_hand_combination(list_cards)).to_single_representation()
