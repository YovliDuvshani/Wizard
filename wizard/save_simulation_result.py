from typing import Optional

from pathlib import Path
import pandas as pd

from wizard.simulation_result import SimulationResultMetadata


class SaveSimulationResult:
    def __init__(
        self,
        simulation_result_metadata: SimulationResultMetadata,
        simulation_result: pd.DataFrame,
        prefix_path: Optional[str] = "",
        suffix_path: Optional[str] = "",
    ):
        self.simulation_result_metadata = simulation_result_metadata
        self.simulation_result = simulation_result
        self.suffix_path = suffix_path
        self.prefix_path = prefix_path

    def save_simulation_result(self):
        path = (
            f"../simulation_result/{self.prefix_path}"
            f"number_players_{self.simulation_result_metadata.number_of_players}/"
            f"number_cards_per_player_{self.simulation_result_metadata.number_of_cards_per_player}/"
            f"learning_player_id_{self.simulation_result_metadata.learning_player_id}/"
            f"{self.suffix_path}{self.simulation_result_metadata.total_number_trial}_trials_"
            f"id_{self.simulation_result_metadata.simulation_id}.csv"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.simulation_result.to_csv(path)
