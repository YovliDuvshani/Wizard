from typing import Optional

import pandas as pd

from config.common import COMBINATION_INDEXES, NUMBER_CARDS_PER_PLAYER
from wizard.base_game.count_points import CountPoints
from wizard.base_game.list_cards import ListCards


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
        return self.sort_per_combination(
            pd.concat(
                [
                    self.evaluate_optimal_strategy_for_given_prediction(
                        evaluated_prediction=prediction
                    )
                    .rename(columns={"score": f"score_prediction_{prediction}"})
                    .set_index(COMBINATION_INDEXES)
                    for prediction in self.evaluated_predictions
                ],
                axis=1,
                join="inner",
            ).reset_index()
        ).set_index(["tested_combination"])

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
                COMBINATION_INDEXES + ["trial_number"]
            )
            .score.min()
            .reset_index()
        )

        mean_worst_outcome_per_order = (
            worst_outcome_per_trial_per_order.groupby(COMBINATION_INDEXES)
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

    @staticmethod
    def sort_per_combination(simulation_results: pd.DataFrame):
        simulation_results["tested_combination_in_card_format"] = simulation_results[
            "tested_combination"
        ].apply(
            lambda representation: ListCards.from_single_representation(
                representation=representation
            ).cards
        )
        return simulation_results.sort_values(
            "tested_combination_in_card_format", ascending=False
        ).drop(columns="tested_combination_in_card_format")


def transform_surveyed_df_to_have_predictions_as_index(
    df: pd.DataFrame,
    number_of_cards_per_player: int = NUMBER_CARDS_PER_PLAYER,
):
    return pd.melt(
        df.rename(
            columns={
                f"score_prediction_{i}": i
                for i in range(number_of_cards_per_player + 1)
            }
        ),
        id_vars=COMBINATION_INDEXES,
        value_vars=range(number_of_cards_per_player + 1),
        var_name="prediction",
        value_name="score",
    ).set_index(COMBINATION_INDEXES + ["prediction"])
