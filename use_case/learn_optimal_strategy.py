import pandas as pd

from config.common import NUMBER_OF_PLAYERS, NUMBER_CARDS_PER_PLAYER
from wizard.deck import Deck
from wizard.player import DefinedStrategyPlayer
from wizard.simulation_result import SimulationResultMetadata
from wizard.simulator import SimulatorWithOneLearningPlayer
from wizard.survey_simulation_result import SurveySimulationResult

NUMBER_TRIALS_EACH_COMBINATION = 10

players = [DefinedStrategyPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
learning_player = players[0]

simulator = SimulatorWithOneLearningPlayer(
    players=players,
    learning_player=learning_player,
    initial_deck=Deck(),
    number_trial_each_combination=NUMBER_TRIALS_EACH_COMBINATION,
)

result_logger = simulator.simulate()

SimulationResultMetadata(
    simulation_id=simulator.simulation_id,
    learning_player_id=learning_player.identifier,
    number_of_players=NUMBER_OF_PLAYERS,
    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
)

survey = SurveySimulationResult(
    simulation_results=pd.DataFrame(result_logger),
    learning_player_id=learning_player.identifier,
    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
)

survey.evaluate_optimal_strategy()
