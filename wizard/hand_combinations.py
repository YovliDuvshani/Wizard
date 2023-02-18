import abc
import itertools
from typing import List

from config.common import JESTER_NAME, MAGICIAN_NAME, TRUMP_COLOR, BASE_COLORS
from wizard.card import Card
from wizard.common import iterator_to_list_of_list
from wizard.deck import Deck


class HandCombinations(abc.ABC):
    def __init__(self, deck: Deck):
        self.deck = deck

    @abc.abstractmethod
    def build_hand_combinations(self):
        pass

    @staticmethod
    def _get_all_subset_size_n(list_cards: List[Card], n: int) -> List[List[Card]]:
        return iterator_to_list_of_list(itertools.combinations(list_cards, n))


# noinspection PyTypeChecker
class HandCombinationsTwoCards(HandCombinations):
    def __init__(self, deck: Deck):
        super().__init__(deck=deck)

    def build_hand_combinations(self) -> List[List[Card]]:
        combinations_same_joker_only = self._get_combinations_two_same_joker()
        combinations_trump_joker = self._get_combinations_trump_joker()
        combinations_one_trump_joker_one_other = (
            self._get_combinations_one_trump_joker_one_other()
        )
        combination_two_others = self._get_combinations_two_others()
        return (
            combinations_same_joker_only
            + combinations_trump_joker
            + combinations_one_trump_joker_one_other
            + combination_two_others
        )

    @staticmethod
    def _get_combinations_two_same_joker() -> List[List[Card]]:
        two_magicians = [
            [Card(color=None, number=None, special_card=MAGICIAN_NAME)] * 2
        ]
        two_jesters = [[Card(color=None, number=None, special_card=JESTER_NAME)] * 2]
        return two_magicians + two_jesters

    def _get_combinations_trump_joker(self) -> List[List[Card]]:
        only_trumps_and_joker = self.deck.filter_deck(
            colors=[TRUMP_COLOR], keep_joker=True, keep_joker_duplicates=False
        )
        return self._get_all_subset_size_n(list_cards=only_trumps_and_joker, n=2)

    def _get_combinations_one_trump_joker_one_other(self) -> List[List[Card]]:
        only_trumps_and_joker = self.deck.filter_deck(
            colors=[TRUMP_COLOR], keep_joker=True, keep_joker_duplicates=False
        )
        only_one_color = self.deck.filter_deck(
            colors=[BASE_COLORS[1]], keep_joker=False, keep_joker_duplicates=False
        )
        return iterator_to_list_of_list(
            itertools.product(only_trumps_and_joker, only_one_color)
        )

    def _get_combinations_two_others(self) -> List[List[Card]]:
        only_one_color = self.deck.filter_deck(
            colors=[BASE_COLORS[1]], keep_joker=False, keep_joker_duplicates=False
        )
        only_one_other_color = self.deck.filter_deck(
            colors=[BASE_COLORS[2]], keep_joker=False, keep_joker_duplicates=False
        )

        all_combinations_one_color_only = self._get_all_subset_size_n(
            list_cards=only_one_color, n=2
        )

        return (
            self._get_unique_pairs_from_two_colors(only_one_color, only_one_other_color)
            + all_combinations_one_color_only
        )

    @staticmethod
    def _get_unique_pairs_from_two_colors(
        one_color: List[Card], other_color: List[Card]
    ) -> List[List[Card]]:
        all_pairs_from_two_colors = iterator_to_list_of_list(
            itertools.product(one_color, other_color)
        )
        return [
            pair
            for pair in all_pairs_from_two_colors
            if pair[0].number >= pair[1].number
        ]
