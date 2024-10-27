import abc

import numpy as np

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.base_game.list_cards import ListCards
from wizard.simulation.exhaustive.hand_combinations import HandCombinationsTwoCards


class BasePredictionPolicy(abc.ABC):
    def __init__(self, player):
        self._player = player

    def _possible_predictions(self) -> list[int]:
        if self._player.game.ordered_list_players[-1] is self:
            sum_of_already_announced_predictions = sum(
                self._player.game.state.predictions[player] for player in self._player.game.ordered_list_players[:-1]
            )
            if (forbidden_prediction := NUMBER_CARDS_PER_PLAYER - sum_of_already_announced_predictions) >= 0:
                return list(range(forbidden_prediction)) + list(
                    range(forbidden_prediction + 1, NUMBER_CARDS_PER_PLAYER + 1)
                )
        return list(range(NUMBER_CARDS_PER_PLAYER + 1))

    @abc.abstractmethod
    def execute(self) -> int:
        pass


class RandomPredictionPolicy(BasePredictionPolicy):
    def execute(self) -> int:
        return np.random.choice(self._possible_predictions())


class DefinedPredictionPolicy(BasePredictionPolicy):
    def execute(self) -> int:
        prediction = self._player.set_prediction
        assert prediction is not None, "No prediction given"
        if prediction in self._possible_predictions():
            return prediction
        return prediction + 1


class StatisticalPredictionPolicy(BasePredictionPolicy):
    def execute(self):
        prediction = self._optimal_strategy.index.get_level_values("prediction")[0]
        if prediction in self._possible_predictions():
            return prediction
        return prediction + 1

    @property
    def _optimal_strategy(self):
        assert self._player.stat_table is not None, "No stat table provided"
        return self._player.stat_table.loc[
            self._player.stat_table[
                self._player.stat_table.index.get_level_values("tested_combination")
                == ListCards(self._initial_hand_combination).to_single_representation()
            ].idxmax(),
            :,
        ]

    @property
    def _initial_hand_combination(self):
        return HandCombinationsTwoCards().list_cards_to_hand_combination(self._player.initial_cards)


class DQNPredictionPolicy(BasePredictionPolicy):
    def execute(self):
        assert self._player.agent is not None, "No DQN agent provided"
        features = self._compute_features()
        best_predictions = self._player.agent.get_highest_rewards_predictions(features)
        if best_predictions[0] in self._possible_predictions():
            return best_predictions[0]
        return best_predictions[1]

    def _compute_features(self):
        from wizard.rl_pipeline.features.compute_generic_features import (
            ComputeGenericFeatures,
        )
        from wizard.rl_pipeline.features.select_learning_features import (
            SelectLearningFeatures,
        )

        return SelectLearningFeatures().execute(ComputeGenericFeatures(self._player.game, self._player).execute())
