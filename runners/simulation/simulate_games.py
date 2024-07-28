import pandas as pd
from pyinstrument import Profiler

from config.common import NUMBER_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.player.player import RandomPlayer, StatisticalPlayer
from wizard.exhaustive_simulation.simulation_result import SimulationResultMetadata
from wizard.exhaustive_simulation.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)
from wizard.exhaustive_simulation.survey_simulation_result import (
    transform_surveyed_df_to_have_predictions_as_index,
)

profiler = Profiler()
profiler.start()

NUMBER_TRIALS = 5000

simulation_ids_per_player = {
    0: 195991601809031998,
    1: 9096319617655858505,
    2: 1527386806013840933,
}
players = [
    StatisticalPlayer(
        identifier=i,
        stat_table=transform_surveyed_df_to_have_predictions_as_index(
            SimulationResultStorage(
                simulation_result_metadata=SimulationResultMetadata(
                    simulation_id=simulation_ids_per_player[i],
                    learning_player_id=i,
                    number_of_players=NUMBER_OF_PLAYERS,
                    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
                    total_number_trial=1000,
                ),
                simulation_type=SimulationResultType.SURVEY,
            ).read_simulation_result()
        ),
    )
    for i in range(3)
]

players = [RandomPlayer(i) for i in range(3)]

games_scoring = []
for _ in range(NUMBER_TRIALS):
    game = Game()
    game.initialize_game(deck=Deck(), players=players, first_player=players[0])
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
