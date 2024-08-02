import pandas as pd

from wizard.simulation.exhaustive.simulation_result import SimulationResultMetadata
from wizard.simulation.exhaustive.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)

simulation_result_metadata = SimulationResultMetadata(
    simulation_id=-2449507750738433667,
    learning_player_id=0,
    number_of_players=3,
    number_of_cards_per_player=2,
    total_number_trial=-1,
)

SimulationResultStorage(
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.ALL_OUTCOME,
).save_simulation_result(pd.DataFrame())

simulation_result = SimulationResultStorage(
    simulation_result_metadata=simulation_result_metadata,
    simulation_type=SimulationResultType.ALL_OUTCOME,
).read_simulation_result()
