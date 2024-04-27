import pandas as pd
from pyinstrument import Profiler

from config.common import NUMBER_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.deck import Deck
from wizard.base_game.player import DefinedStrategyPlayer
from wizard.exhaustive_simulation.simulation_result import SimulationResultMetadata
from wizard.exhaustive_simulation.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)
from wizard.exhaustive_simulation.simulator import SimulatorWithOneLearningPlayer
from wizard.exhaustive_simulation.survey_simulation_result import SurveySimulationResult

# profiler = Profiler()
# profiler.start()

NUMBER_TRIALS_EACH_COMBINATION = 10

players = [DefinedStrategyPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
learning_player = players[1]

simulator = SimulatorWithOneLearningPlayer(
    players=players,
    learning_player=learning_player,
    initial_deck=Deck(),
    number_trial_each_combination=NUMBER_TRIALS_EACH_COMBINATION,
)

simulation_result_metadata = SimulationResultMetadata(
    simulation_id=simulator.simulation_id,
    learning_player_id=learning_player.identifier,
    number_of_players=NUMBER_OF_PLAYERS,
    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
    total_number_trial=NUMBER_TRIALS_EACH_COMBINATION,
)
simulation_result = pd.DataFrame(simulator.simulate())

SimulationResultStorage(
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.ALL_OUTCOME,
).save_simulation_result(simulation_result)


survey = SurveySimulationResult(
    simulation_results=simulation_result,
    learning_player_id=learning_player.identifier,
    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
)
surveyed_simulation_result = survey.evaluate_optimal_strategy()

SimulationResultStorage(
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.SURVEY,
).save_simulation_result(surveyed_simulation_result)

# profiler.stop()
