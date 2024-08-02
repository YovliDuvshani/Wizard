from copy import deepcopy

import numpy as np

from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.player.player import Player


class SimulatePreDefinedGames:
    def __init__(self, num_games: int):
        self._decks = [Deck() for _ in range(num_games)]
        self._players = None
        self._target_player = None
        self._player_starting = None

    def assign_players(self, players: list[Player], target_player: Player, player_starting: Player):
        self._players = players
        self._target_player = target_player
        self._player_starting = player_starting

    def execute(self) -> float:
        rewards = []
        decks = deepcopy(self._decks)
        for deck in decks:
            game = Game()
            game.initialize_game(deck=deck, players=self._players, first_player=self._player_starting)
            game.request_predictions()
            game.play_game()
            rewards.append(
                CountPoints().execute(game.state.predictions, game.state.number_of_turns_won)[
                    self._target_player.identifier
                ]
            )
        return float(np.mean(rewards))
