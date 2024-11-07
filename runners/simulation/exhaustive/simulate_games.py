import pandas as pd
from pyinstrument import Profiler

from config.common import NUMBER_OF_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.player.player import RandomPlayer, StatisticalPlayer
from wizard.simulation.exhaustive.simulation_result import SimulationResultMetadata
from wizard.simulation.exhaustive.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)

profiler = Profiler()
profiler.start()

NUMBER_TRIALS = 50

players = [StatisticalPlayer(identifier=i) for i in range(3)]

games_scoring = []
for _ in range(NUMBER_TRIALS):
    game = Game()
    game.initialize_game(deck=Deck(), players=players, starting_player=players[0])
    game.request_predictions()
    game.play_game()

    games_scoring.append(
        CountPoints().execute(
            predictions=game.state.predictions,
            number_of_turns_won=game.state.number_of_turns_won,
        )
    )
games_scoring_pd = pd.DataFrame(games_scoring)

profiler.stop()
profiler.open_in_browser()
