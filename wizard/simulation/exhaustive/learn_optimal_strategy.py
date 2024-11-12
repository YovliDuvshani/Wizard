import pandas as pd

from config.common import NUMBER_OF_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.deck import Deck
from wizard.base_game.player.player import DefinedStrategyPlayer
from wizard.simulation.exhaustive.simulation_result import SimulationResultMetadata
from wizard.simulation.exhaustive.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)
from wizard.simulation.exhaustive.simulator import SimulatorWithOneLearningPlayer
from wizard.simulation.exhaustive.survey_simulation_result import SurveySimulationResult

# profiler = Profiler()
# profiler.start()

NUMBER_TRIALS_EACH_COMBINATION = 500

players = [DefinedStrategyPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
learning_player = players[2]

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
    number_of_cards_per_player=NUMBER_OF_CARDS_PER_PLAYER,
    total_number_trial=NUMBER_TRIALS_EACH_COMBINATION,
)
simulation_result = pd.DataFrame(simulator.simulate())

SimulationResultStorage().save_simulation_result(
    simulation_result=simulation_result,
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.ALL_OUTCOME,
)


survey = SurveySimulationResult(
    simulation_results=simulation_result,
    learning_player_id=learning_player.identifier,
    number_of_cards_per_player=NUMBER_OF_CARDS_PER_PLAYER,
)
surveyed_simulation_result = survey.compute_optimal_strategy()

SimulationResultStorage().save_simulation_result(
    simulation_result=surveyed_simulation_result,
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.SURVEY,
)

# profiler.stop()
