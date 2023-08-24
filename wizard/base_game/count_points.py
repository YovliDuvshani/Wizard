from typing import Dict, Union

from config.common import BASE_REWARD, DYNAMIC_LOSS, DYNAMIC_REWARD
from wizard.base_game.player import Player


class CountPoints:
    @staticmethod
    def count_points_single_prediction(prediction: int, number_of_turns_won: int):
        if prediction == number_of_turns_won:
            return int((DYNAMIC_REWARD * prediction + BASE_REWARD))
        return int(DYNAMIC_LOSS * abs(prediction - number_of_turns_won))

    def count_points_round(
        self,
        predictions: Dict[Player, int],
        number_of_turns_won: Dict[Union[int, Player], int],
    ) -> Dict[int, int]:
        score: Dict[int, int] = {}
        for player in predictions:
            score[player.identifier] = self.count_points_single_prediction(
                prediction=predictions[player],
                number_of_turns_won=number_of_turns_won[player],
            )
        return score
