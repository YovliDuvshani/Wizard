from dataclasses import dataclass
from typing import Dict


@dataclass
class SimulationResult:
    trial_number: int
    tested_combination: str
    combination_played_order: str
    number_of_turns_won: Dict[int, int]


@dataclass
class SimulationResultMetadata:
    simulation_id: int
    learning_player_id: int
    number_of_players: int
    number_of_cards_per_player: int
    total_number_trial: int
