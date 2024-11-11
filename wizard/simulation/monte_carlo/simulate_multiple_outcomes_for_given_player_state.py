from wizard.base_game.game import Game
from wizard.base_game.player.player import Player


class GameMustHaveBeenInitialized(Exception):
    pass


class SimulateMultipleOutcomesForGivenPlayerState:
    def __init__(self, game: Game, learning_player: Player):
        if game.definition is None:
            raise GameMustHaveBeenInitialized
        self._game = game
        self._learning_player = learning_player
