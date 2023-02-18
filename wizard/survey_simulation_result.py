from typing import Optional

import pandas as pd

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.count_points import CountPoints


class SurveySimulationResult:
    def __init__(
        self,
        simulation_results: pd.DataFrame,
        learning_player_id: int,
        number_of_cards_per_player: Optional[int] = NUMBER_CARDS_PER_PLAYER,
    ):
        self.learning_player_id = learning_player_id
        self.simulation_results = simulation_results
        self.number_of_cards_per_player = number_of_cards_per_player

    def evaluate_optimal_strategy(self):
        mean_worst_outcome_per_order_each_prediction = pd.DataFrame
        for prediction in self.evaluated_predictions:
            mean_worst_outcome_per_order_given_prediction = (
                self.evaluate_optimal_strategy_for_given_prediction(
                    evaluated_prediction=prediction
                ).rename(columns={"score": f"score_prediction_{prediction}"})
            )
            if mean_worst_outcome_per_order_each_prediction.empty:
                mean_worst_outcome_per_order_each_prediction = (
                    mean_worst_outcome_per_order_given_prediction
                )
            else:
                mean_worst_outcome_per_order_each_prediction = (
                    mean_worst_outcome_per_order_each_prediction.merge(
                        mean_worst_outcome_per_order_given_prediction,
                        on=["tested_combination", "combination_played_order"],
                    )
                )
        return mean_worst_outcome_per_order_each_prediction

    @property
    def evaluated_predictions(self):
        return list(range(self.number_of_cards_per_player + 1))

    def evaluate_optimal_strategy_for_given_prediction(self, evaluated_prediction: int):
        simulation_results_with_score = (
            self.simulation_results_with_score_for_given_prediction(
                prediction=evaluated_prediction
            )
        )
        worst_outcome_per_trial_per_order = (
            simulation_results_with_score.groupby(
                ["tested_combination", "combination_played_order", "trial_number"]
            )
            .score.min()
            .reset_index()
        )

        mean_worst_outcome_per_order = (
            worst_outcome_per_trial_per_order.groupby(
                ["tested_combination", "combination_played_order"]
            )
            .score.mean()
            .reset_index()
        )

        return mean_worst_outcome_per_order

    def simulation_results_with_score_for_given_prediction(self, prediction: int):
        simulation_results = self.simulation_results.copy()
        simulation_results["score"] = simulation_results["number_of_turns_won"].apply(
            lambda row: CountPoints().count_points_single_prediction(
                prediction=prediction, number_of_turns_won=row[self.learning_player_id]
            ),
        )

        return simulation_results
