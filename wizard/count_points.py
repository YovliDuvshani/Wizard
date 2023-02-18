from typing import Dict, Union

from config.common import DYNAMIC_REWARD, BASE_REWARD, DYNAMIC_LOSS
from wizard.player import Player


class CountPoints:
    @staticmethod
    def count_points_single_prediction(prediction: int, number_of_turns_won: int):
        if prediction == number_of_turns_won:
            return int((DYNAMIC_REWARD * prediction + BASE_REWARD))
        return int(DYNAMIC_LOSS * abs(prediction - number_of_turns_won))

    def count_points_round(
        self,
        predictions: Dict[Union[int, Player], int],
        number_of_turns_won: Dict[Union[int, Player], int],
    ):
        score: Dict[Union[int, Player], int] = {}
        for identifier in predictions.keys():
            score[identifier] = self.count_points_single_prediction(
                prediction=predictions[identifier],
                number_of_turns_won=number_of_turns_won[identifier],
            )
        return score
