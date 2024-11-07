from typing import Dict, Union

from config.common import (
    BASE_REWARD,
    DYNAMIC_LOSS,
    DYNAMIC_REWARD,
    NUMBER_OF_CARDS_PER_PLAYER,
)
from wizard.base_game.player.player import Player

POINT_RANGE = (
    -DYNAMIC_REWARD * NUMBER_OF_CARDS_PER_PLAYER,
    DYNAMIC_REWARD * NUMBER_OF_CARDS_PER_PLAYER + BASE_REWARD,
)


class CountPoints:
    def execute(
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

    @staticmethod
    def count_points_single_prediction(prediction: int, number_of_turns_won: int):
        if prediction == number_of_turns_won:
            return int((DYNAMIC_REWARD * prediction + BASE_REWARD))
        return int(DYNAMIC_LOSS * abs(prediction - number_of_turns_won))
