from copy import deepcopy

import numpy as np

from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.player.player import Player


class SimulatePreDefinedGames:
    def __init__(
        self, decks: list[Deck], players: list[Player], learning_player: Player, starting_player: Player | None
    ):
        self._decks = deepcopy(decks)
        self._players = players
        self._learning_player = learning_player
        self._starting_player = starting_player

    def execute(self) -> float:
        rewards = []
        for i, deck in enumerate(self._decks):
            starting_player = (
                self._starting_player if self._starting_player else self._players[i % NUMBER_OF_PLAYERS]
            )  # Ensures no changes between different runs for static policies
            game = Game()
            game.initialize_game(deck=deck, players=self._players, starting_player=starting_player, deterministic=True)
            game.request_predictions()
            game.play_game()
            rewards.append(
                CountPoints().execute(game.state.predictions, game.state.number_of_turns_won)[
                    self._learning_player.identifier
                ]
            )
        return float(np.mean(rewards))
