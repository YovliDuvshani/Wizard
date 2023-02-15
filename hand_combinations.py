from card import Card
from deck import Deck


class HandCombinations:
    def __init__(self, trump_color):
        self.trump_color = trump_color

    def possibilities_to_test(self, deck: Deck):
        pass


# noinspection PyTypeChecker
class HandCombinationsTwoCards(HandCombinations):
    def __init__(self, trump_color, base_color1, base_color2):
        super().__init__(trump_color)
        self.base_color1 = base_color1
        self.base_color2 = base_color2

    def possibilities_to_test(self, deck: Deck):
        combinations_same_joker_only = self._get_combinations_two_same_joker()
        combinations_trump_joker = self._get_combinations_trump_joker(deck)
        combinations_one_trump_joker_one_other = self._get_combinations_one_trump_joker_one_other(deck)
        combination_two_others = self._get_combinations_two_others(deck)
        return combinations_same_joker_only + combinations_trump_joker + combinations_one_trump_joker_one_other + combination_two_others

    @staticmethod
    def _get_combinations_two_same_joker():
        two_wizards = [[Card(color=None, number=None, special_card="Wizard"),
                        Card(color=None, number=None, special_card="Wizard")]]
        two_jester = [[Card(color=None, number=None, special_card="Jester"),
                       Card(color=None, number=None, special_card="Jester")]]
        return two_wizards + two_jester

    def _get_combinations_trump_joker(self, deck):
        cards_filtered = deck.filter_deck(colors=[self.trump_color],
                                          add_joker=True,
                                          keep_joker_duplicates=False)
        return self.get_all_subset_size_two(cards_filtered)

    def _get_combinations_one_trump_joker_one_other(self, deck):
        cards_filtered_1 = deck.filter_deck(colors=[self.trump_color],
                                            add_joker=True,
                                            keep_joker_duplicates=False)
        cards_filtered_2 = deck.filter_deck(colors=[self.base_color1],
                                            add_joker=False,
                                            keep_joker_duplicates=False)
        return self.get_all_subset_size_two_from_two_disjoint_list(cards_filtered_1, cards_filtered_2)

    def _get_combinations_two_others(self, deck):
        cards_filtered_1 = deck.filter_deck(colors=[self.base_color1],
                                            add_joker=False,
                                            keep_joker_duplicates=False)
        cards_filtered_2 = deck.filter_deck(colors=[self.base_color2],
                                            add_joker=False,
                                            keep_joker_duplicates=False)
        all_pairs_from_two_colors = self.get_all_subset_size_two_from_two_disjoint_list(cards_filtered_1,
                                                                                        cards_filtered_2)
        unique_pairs_from_two_colors = self.retrieve_only_unique_pairs(all_pairs_from_two_colors)
        return unique_pairs_from_two_colors + self.get_all_subset_size_two(cards_filtered_1)

    @staticmethod
    def get_all_subset_size_two(list_cards):
        all_subset = []
        for ind, first_card in enumerate(list_cards):
            for second_card in list_cards[(ind + 1):]:
                all_subset += [[first_card, second_card]]
        return all_subset

    @staticmethod
    def get_all_subset_size_two_from_two_disjoint_list(list_cards1, list_cards2):
        all_subset = []
        for first_card in list_cards1:
            for second_card in list_cards2:
                all_subset += [[first_card, second_card]]
        return all_subset

    @staticmethod
    def retrieve_only_unique_pairs(list_cards):
        unique_pairs_list = []
        for pair in list_cards:
            if pair[0].number >= pair[1].number:
                unique_pairs_list += [pair]
        return unique_pairs_list
