import os
from enum import Enum
from pathlib import Path

import pandas as pd

from config.common import NUMBER_OF_PLAYERS, NUMBER_OF_CARDS_PER_PLAYER
from project_path import ABS_PATH_PROJECT
from wizard.simulation.exhaustive.constants import COMBINATION_INDEXES
from wizard.simulation.exhaustive.simulation_result import SimulationResultMetadata


class SimulationResultType(Enum):
    ALL_OUTCOME = "all_outcome"
    SURVEY = "survey"


class SimulationResultStorage:
    BASE_PATH = f"{ABS_PATH_PROJECT}/simulation_result/"

    def save_simulation_result(
        self,
        simulation_result: pd.DataFrame,
        simulation_result_metadata: SimulationResultMetadata,
        simulation_type: SimulationResultType,
    ):
        path = self._get_path(simulation_result_metadata, simulation_type)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        simulation_result.to_csv(path)

    def read_given_simulation_result(
        self, simulation_result_metadata: SimulationResultMetadata, simulation_type: SimulationResultType
    ):
        return pd.read_csv(self._get_path(simulation_result_metadata, simulation_type))

    def _get_path(self, simulation_result_metadata: SimulationResultMetadata, simulation_type: SimulationResultType):
        return (
            f"{self.BASE_PATH}"
            f"number_of_players={simulation_result_metadata.number_of_players}/"
            f"number_cards_per_player={simulation_result_metadata.number_of_cards_per_player}/"
            f"learning_player_position={simulation_result_metadata.learning_player_id}/"
            f"{simulation_type.value}/"
            f"{simulation_result_metadata.total_number_trial}_trials_"
            f"id_{simulation_result_metadata.simulation_id}.csv"
        )

    def read_surveyed_simulation_result_based_on_current_configuration(self, player_position: int):
        folder_path = (
            f"{self.BASE_PATH}"
            f"number_of_players={NUMBER_OF_PLAYERS}/"
            f"number_cards_per_player={NUMBER_OF_CARDS_PER_PLAYER}/"
            f"learning_player_position={player_position}/"
            f"{SimulationResultType.SURVEY.value}/"
        )
        file_paths = os.listdir(folder_path)
        selected_file_path = max(file_paths, key=lambda file_path: int(file_path.split("_")[0]))
        return self._set_surveyed_df_predictions_as_index(pd.read_csv(folder_path + selected_file_path))

    @staticmethod
    def _set_surveyed_df_predictions_as_index(df: pd.DataFrame):
        return pd.melt(
            df.rename(columns={f"score_prediction_{i}": i for i in range(NUMBER_OF_CARDS_PER_PLAYER + 1)}),
            id_vars=COMBINATION_INDEXES,
            value_vars=range(NUMBER_OF_CARDS_PER_PLAYER + 1),
            var_name="prediction",
            value_name="score",
        ).set_index(COMBINATION_INDEXES + ["prediction"])
