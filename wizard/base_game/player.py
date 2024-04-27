# mypy: disable-error-code="attr-defined"
import abc
from typing import List, Optional

import numpy as np
import pandas as pd

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.base_game.card import Card
from wizard.base_game.list_cards import ListCards
from wizard.exhaustive_simulation.hand_combinations import HandCombinationsTwoCards


class Player(abc.ABC):
    def __init__(self, identifier: int):
        self.identifier = identifier
        self.cards: Optional[List[Card]] = None
        self.initial_cards: Optional[List[Card]] = None
        self.game = None

    def assign_game(self, game):
        self.game = game

    def receive_cards(self, cards: List[Card]):
        if not self.cards:
            self.cards = cards
            self.initial_cards = cards.copy()
            return True
        return False

    def _filter_playable_cards_relatively_to_first_color_played(
        self, first_color: str
    ) -> List[Card]:
        cards_from_required_color: List[Card] = []
        special_cards: List[Card] = []
        for card in self.cards:
            if card.color == first_color:
                cards_from_required_color += [card]
            elif card.color is None:
                special_cards += [card]
        if cards_from_required_color:
            return cards_from_required_color + special_cards
        return self.cards

    def playable_cards(self) -> List[Card]:
        if self.game.state.current_turn_history:
            first_color_played = self.game.state.current_turn_history[-1].starting_color
            if first_color_played is not None:
                return self._filter_playable_cards_relatively_to_first_color_played(
                    first_color_played
                )
        return self.cards

    def _possible_predictions(self) -> List[int]:
        if self.game.ordered_list_players[-1] == self:
            sum_of_already_announced_predictions = sum(
                self.game.state.predictions[player]
                for player in self.game.ordered_list_players[:-1]
            )
            if (
                forbidden_prediction := NUMBER_CARDS_PER_PLAYER
                - sum_of_already_announced_predictions
            ) >= 0:
                return list(range(forbidden_prediction)) + list(
                    range(forbidden_prediction + 1, NUMBER_CARDS_PER_PLAYER + 1)
                )
        return list(range(NUMBER_CARDS_PER_PLAYER + 1))

    def reset_hand(self) -> None:
        self.cards = self.initial_cards.copy()

    @abc.abstractmethod
    def make_prediction(self) -> int:
        pass

    @abc.abstractmethod
    def play_card(self) -> Card:
        """
        Play a card during a turn following a strategy.
        The card needs to be returned and removed from the hand of the Player.
        :return: Card that is played
        """
        pass


class RandomPlayer(Player):
    def make_prediction(self) -> int:
        return np.random.choice(self._possible_predictions())

    def play_card(self) -> Card:
        card_to_play = np.random.choice(self.playable_cards())  # type: ignore
        self.cards.remove(card_to_play)
        return card_to_play


class DefinedStrategyPlayer(Player):
    def __init__(self, identifier: int):
        super().__init__(identifier=identifier)
        self.cards_ordered_by_priority: Optional[List[Card]] = None
        self.prediction: Optional[int] = None

    def provide_strategy(
        self,
        cards_ordered_by_priority: Optional[List[Card]] = None,
        prediction: Optional[int] = None,
    ):
        self.cards_ordered_by_priority = cards_ordered_by_priority
        self.prediction = prediction

    def play_card(self) -> Card:  # type: ignore
        assert self.cards_ordered_by_priority, "No priority given"
        for card in self.cards_ordered_by_priority:
            if card in self.playable_cards():
                self.cards.remove(card)
                return card

    def make_prediction(self) -> int:
        assert self.prediction, "No prediction given"
        if self.prediction in self._possible_predictions():
            return self.prediction
        return self.prediction + 1


class StatisticalPlayer(Player):
    def __init__(self, identifier: int, stat_table: pd.DataFrame):
        super().__init__(identifier)
        self.stat_table = stat_table

    def make_prediction(self):
        prediction = self._optimal_strategy.index.get_level_values("prediction")[0]
        if prediction in self._possible_predictions():
            return prediction
        return prediction + 1

    def play_card(self):
        for card in self.cards_ordered_by_priority:
            if card in self.playable_cards():
                self.cards.remove(card)
                return card

    @property
    def cards_ordered_by_priority(self) -> List[Card]:
        cards_ordered_by_priority = []
        for placeholder_order_card in ListCards.from_single_representation(
            self._optimal_strategy.index.get_level_values("combination_played_order")[0]
        ).cards:
            for ind, placeholder_combination_card in enumerate(
                self._initial_hand_combination
            ):
                if placeholder_order_card == placeholder_combination_card:
                    cards_ordered_by_priority.append(self.initial_cards[ind])
                    break
        return cards_ordered_by_priority

    @property
    def _optimal_strategy(self):
        return self.stat_table.loc[
            self.stat_table[
                self.stat_table.index.get_level_values("tested_combination")
                == ListCards(self._initial_hand_combination).to_single_representation()
            ].idxmax(),
            :,
        ]

    @property
    def _initial_hand_combination(self):
        return HandCombinationsTwoCards().list_cards_to_hand_combination(
            self.initial_cards
        )
