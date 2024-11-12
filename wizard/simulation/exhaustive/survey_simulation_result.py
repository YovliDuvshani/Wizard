from typing import Optional

import pandas as pd

from config.common import NUMBER_OF_CARDS_PER_PLAYER
from wizard.base_game.count_points import CountPoints
from wizard.base_game.hand import Hand
from wizard.simulation.exhaustive.constants import COMBINATION_INDEXES


class SurveySimulationResult:
    def __init__(
        self,
        simulation_results: pd.DataFrame,
        learning_player_id: int,
        number_of_cards_per_player: Optional[int] = NUMBER_OF_CARDS_PER_PLAYER,
    ):
        self.learning_player_id = learning_player_id
        self.simulation_results = simulation_results
        self.number_of_cards_per_player = number_of_cards_per_player

    def compute_optimal_strategy(self):
        return self._sort_per_combination(
            pd.concat(
                [
                    self._evaluate_optimal_strategy_for_given_prediction(evaluated_prediction=prediction)
                    .rename(columns={"score": f"score_prediction_{prediction}"})
                    .set_index(COMBINATION_INDEXES)
                    for prediction in self._evaluated_predictions
                ],
                axis=1,
                join="inner",
            ).reset_index()
        ).set_index(["tested_combination"])

    def _evaluate_optimal_strategy_for_given_prediction(self, evaluated_prediction: int):
        simulation_results_with_score = self._simulation_results_with_score_for_given_prediction(
            prediction=evaluated_prediction
        )
        worst_outcome_per_trial_per_order = (
            simulation_results_with_score.groupby(COMBINATION_INDEXES + ["trial_number"]).score.min().reset_index()
        )

        mean_worst_outcome_per_order = (
            worst_outcome_per_trial_per_order.groupby(COMBINATION_INDEXES).score.mean().reset_index()
        )

        return mean_worst_outcome_per_order

    def _simulation_results_with_score_for_given_prediction(self, prediction: int):
        simulation_results = self.simulation_results.copy()
        simulation_results["score"] = simulation_results["number_of_turns_won"].apply(
            lambda row: CountPoints().count_points_single_prediction(
                prediction=prediction, number_of_turns_won=row[self.learning_player_id]
            ),
        )
        return simulation_results

    @staticmethod
    def _sort_per_combination(simulation_results: pd.DataFrame):
        simulation_results["tested_combination_in_card_format"] = simulation_results["tested_combination"].apply(
            lambda representation: Hand.from_single_representation(representation=representation).cards
        )
        return simulation_results.sort_values("tested_combination_in_card_format", ascending=False).drop(
            columns="tested_combination_in_card_format"
        )

    @property
    def _evaluated_predictions(self):
        return list(range(self.number_of_cards_per_player + 1))