from enum import Enum
from pathlib import Path

import pandas as pd

from project_path import ABS_PATH_PROJECT
from wizard.exhaustive_simulation.simulation_result import \
    SimulationResultMetadata


class SimulationResultType(Enum):
    ALL_OUTCOME = "all_outcome"
    SURVEY = "survey"


class SimulationResultStorage:
    def __init__(
        self,
        simulation_result_metadata: SimulationResultMetadata,
        simulation_type: SimulationResultType,
    ):
        self.simulation_result_metadata = simulation_result_metadata
        self.simulation_type = simulation_type

    def save_simulation_result(self, simulation_result: pd.DataFrame):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        simulation_result.to_csv(self.path)

    def read_simulation_result(self):
        return pd.read_csv(self.path)

    @property
    def path(self):
        return (
            f"{ABS_PATH_PROJECT}/simulation_result/"
            f"number_players_{self.simulation_result_metadata.number_of_players}/"
            f"number_cards_per_player_{self.simulation_result_metadata.number_of_cards_per_player}/"
            f"learning_player_id_{self.simulation_result_metadata.learning_player_id}/"
            f"{self.simulation_type.value}/"
            f"{self.simulation_result_metadata.total_number_trial}_trials_"
            f"id_{self.simulation_result_metadata.simulation_id}.csv"
        )
