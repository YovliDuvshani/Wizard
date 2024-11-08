import abc

import numpy as np

from config.common import NUMBER_OF_CARDS_PER_PLAYER
from wizard.base_game.card import Card
from wizard.base_game.list_cards import ListCards
from wizard.simulation.exhaustive.hand_combinations import HandCombinationsTwoCards, IMPLEMENTED_COMBINATIONS
from wizard.simulation.exhaustive.simulation_result_storage import SimulationResultStorage


class BaseCardPlayPolicy(abc.ABC):
    def __init__(self, player):
        self._player = player

    def playable_cards(self) -> list[Card]:
        first_color_played = self._player.game.state.round_specifics.starting_color
        if first_color_played is not None:
            return self._filter_playable_cards_relatively_to_first_color_played(first_color_played)
        return self._player.cards

    def _filter_playable_cards_relatively_to_first_color_played(self, first_color: str) -> list[Card]:
        cards_from_required_color: list[Card] = []
        special_cards: list[Card] = []
        for card in self._player.cards:
            if card.color == first_color:
                cards_from_required_color += [card]
            elif card.color is None:
                special_cards += [card]
        if cards_from_required_color:
            return cards_from_required_color + special_cards
        return self._player.cards

    @abc.abstractmethod
    def execute(self) -> Card:
        """
        Returns the card to play during a turn according to a policy.
        :return: Card that is played
        """
        pass


class RandomCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        return np.random.choice(self.playable_cards())  # type: ignore


class HighestCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        return max(self.playable_cards())


class LowestCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        return min(self.playable_cards())


class DefinedCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        assert self._player.set_card_play_priority is not None, "No card priority given"
        for card in self._player.set_card_play_priority:
            if card in self.playable_cards():
                return card


class StatisticalCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        for card in self.cards_ordered_by_priority:
            if card in self.playable_cards():
                return card

    @property
    def cards_ordered_by_priority(self) -> list[Card]:
        cards_ordered_by_priority = []
        for placeholder_order_card in ListCards.from_single_representation(
            self._optimal_strategy.index.get_level_values("combination_played_order")[0]
        ).cards:
            for ind, placeholder_combination_card in enumerate(self._initial_hand_combination):
                if placeholder_order_card == placeholder_combination_card:
                    cards_ordered_by_priority.append(self._player.initial_cards[ind])
                    break
        return cards_ordered_by_priority

    @property
    def _optimal_strategy(self):
        return self._adequate_surveyed_simulation_result.loc[
            self._adequate_surveyed_simulation_result[
                self._adequate_surveyed_simulation_result.index.get_level_values("tested_combination")
                == ListCards(self._initial_hand_combination).to_single_representation()
            ].idxmax(),
            :,
        ]

    @property
    def _initial_hand_combination(self):
        hand_combination_cls = IMPLEMENTED_COMBINATIONS[NUMBER_OF_CARDS_PER_PLAYER]
        return hand_combination_cls().list_cards_to_hand_combination(self._player.initial_cards)

    @property
    def _adequate_surveyed_simulation_result(self):
        return SimulationResultStorage().read_surveyed_simulation_result_based_on_current_configuration(
            self._player.position
        )


class DQNCardPlayPolicy(BaseCardPlayPolicy):
    def execute(self) -> Card:
        features = self._compute_features()
        action = self._player.agent.select_action(features)
        card_to_play = [card for card in self._player.cards if card.representation == action][0]
        return card_to_play

    def _compute_features(self):
        from wizard.rl_pipeline.features.compute_generic_features import (
            ComputeGenericFeatures,
        )

        return ComputeGenericFeatures(self._player.game, self._player).execute()
