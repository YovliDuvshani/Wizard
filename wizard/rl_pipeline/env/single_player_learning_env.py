import random
from typing import Any

from gymnasium import Env
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Discrete

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.count_points import POINT_RANGE, CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game, Terminal
from wizard.base_game.player.player import Player
from wizard.rl_pipeline.features.compute_generic_features import ComputeGenericFeatures
from wizard.rl_pipeline.features.observation_space import OBSERVATION_SPACE
from wizard.rl_pipeline.features.select_learning_features import SelectLearningFeatures


class SinglePlayerLearningEnv(Env):
    def __init__(self, players: list[Player], learning_player: Player, starting_player: Player | None = None):
        self.reward_range = POINT_RANGE
        self.action_space = Discrete(NUMBER_OF_UNIQUE_CARDS)
        self.observation_space = OBSERVATION_SPACE

        self._players = players
        self._learning_player = learning_player
        self._starting_player = starting_player

        self._game: Game | None = None

    def step(self, action: ActType) -> tuple[OBSERVATION_SPACE, int, bool, bool, dict[str, Any]]:
        terminal = self._game.get_to_next_afterstate_for_given_player(self._learning_player)
        reward = self._get_reward(terminal)
        return (
            SelectLearningFeatures().execute(ComputeGenericFeatures(self._game, self._learning_player).execute()),
            reward,
            terminal,
            False,
            {},
        )

    def _get_reward(self, terminal: Terminal) -> int:
        if terminal:
            return CountPoints().execute(self._game.state.predictions, self._game.state.number_of_turns_won)[
                self._learning_player.identifier
            ]
        return 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSERVATION_SPACE, dict[str, Any]]:
        self._game = Game()
        for player in self._players:  # Safety mechanism in case of multiple consecutive reset calls
            player.drop_hand()
        self._game.initialize_game(deck=Deck(), players=self._players, starting_player=self._starting_player)
        self._game.request_predictions()
        self._game.get_to_first_state_for_given_player(self._learning_player)
        return (
            SelectLearningFeatures().execute(ComputeGenericFeatures(self._game, self._learning_player).execute()),
            {},
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        pass
