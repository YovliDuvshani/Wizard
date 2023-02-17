import abc
from typing import List

import numpy as np

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.card import Card, Color


class Player(abc.ABC):
    def __init__(self, identifier: int):
        self.cards: List[Card] = []
        self.initial_cards: List[Card] = []
        self.game = None
        self.identifier = identifier

    def assign_game(self, game: object):
        self.game = game

    def receive_cards(self, cards: List[Card]):
        self.cards = cards
        self.initial_cards = cards.copy()

    def _filter_playable_cards_relatively_to_first_color_played(
        self, first_color: Color
    ) -> List[Card]:
        cards_from_required_color: List[Card] = []
        special_cards: List[Card] = []
        for card in self.cards:
            if card.color == first_color:
                cards_from_required_color += [card]
                if card.color is None:
                    special_cards += [card]
        if cards_from_required_color:
            return cards_from_required_color + special_cards
        else:
            return self.cards

    def _playable_cards(self) -> List[Card]:
        if self.game.current_turn_history:
            first_color_played = self.game.current_turn_history[-1].starting_color
            if first_color_played is not None:
                return self._filter_playable_cards_relatively_to_first_color_played(
                    first_color_played
                )
        return self.cards

    def _possible_predictions(self) -> List[int]:
        if self.game.ordered_list_players[-1] == self:
            sum_of_already_announced_predictions = sum(
                [
                    self.game.initial_predictions[player]
                    for player in self.game.initial_predictions.keys()
                ]
            )
            if (
                forbidden_prediction := NUMBER_CARDS_PER_PLAYER
                - sum_of_already_announced_predictions
            ) >= 0:
                return list(range(forbidden_prediction)) + list(
                    range(forbidden_prediction + 1, NUMBER_CARDS_PER_PLAYER + 1)
                )
        return list(range(NUMBER_CARDS_PER_PLAYER + 1))

    def make_prediction(self) -> int:
        pass

    def play_card(self) -> Card:
        """
        Play a card during a turn following a strategy.
        The card both needs to be played and be removed from the hand of the Player.
        :return: Card that is played
        """
        pass


class RandomPlayer(Player):
    def make_prediction(self) -> int:
        return np.random.choice(self._possible_predictions())

    def play_card(self) -> Card:
        card_to_play = np.random.choice(self._playable_cards())
        self.cards.remove(card_to_play)
        return card_to_play


class PreDefinedStrategyPlayer(Player):
    def __init__(
        self, identifier: int, cards_ordered_by_priority: list[Card], prediction: int
    ):
        super().__init__(identifier=identifier)
        self.cards_ordered_by_priority = cards_ordered_by_priority
        self.prediction = prediction

    def play_card(self) -> Card:
        for card in self.cards_ordered_by_priority:
            if card in self._playable_cards():
                self.cards.remove(card)
                return card

    def make_prediction(self) -> int:
        if self.prediction in self._possible_predictions():
            return self.prediction
        return self.prediction + 1


"""

class StatisticPlayer(Player):
    def __init__(
        self, id: str, readable_stat_table: ReadableStatTable, use_base_strategy=True
    ):
        super().__init__(id)
        self.readable_stat_table = readable_stat_table
        self.use_base_strategy = use_base_strategy

    def make_prediction(self, column_best_prediction: str = "best_prediction"):
        stats_for_hand = self.readable_stat_table.get_info_hand(self.cards)
        return stats_for_hand[column_best_prediction]

    def play_card(self, column_priority_plays="Priority_plays"):
        if self.use_base_strategy:
            stats_for_hand = self.readable_stat_table.get_info_hand(self.initial_cards)
            priority_cards_to_play = stats_for_hand[column_priority_plays]
            playable_cards = self._playable_cards()
            for card in priority_cards_to_play:
                if card.is_in_list(playable_cards):
                    card_to_play = card
                    break

        else:  # Not implemented yet
            pass

        self.cards.remove(card_to_play)
        return card_to_play


class RealPlayer(Player):
    def __init__(self, id: str):
        super().__init__(id)

    def make_prediction(self):
        # Print Blabla
        prediction = input("Make a prediction: ")
        return prediction

    def play_card(self):
        # Print Blabla
        playable_cards = self._playable_cards()
        card_indices_to_play = input(
            "Input the indices of the card you want to play among the playable cards: "
        )
        return playable_cards[card_indices_to_play]
"""
