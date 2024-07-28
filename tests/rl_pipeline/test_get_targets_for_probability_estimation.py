from config.common import BASE_COLORS, MAGICIAN_NAME
from wizard.base_game.card import Card
from wizard.base_game.played_card import PlayedCard
from wizard.base_game.player.player import RandomPlayer
from wizard.rl_pipeline.features.get_targets_for_probability_estimation import (
    GetTargetsForProbabilityEstimation,
)


class TestGetTargetsForProbabilityEstimation:
    def test_execute_returns_is_correct(self):
        player = RandomPlayer(identifier=0)
        playable_cards_per_round = [
            [
                Card(color=BASE_COLORS[2], number=2),
                Card(color=BASE_COLORS[2], number=13),
            ],
            [Card(color=BASE_COLORS[2], number=13)],
        ]
        winner_per_round = [
            PlayedCard(
                card=Card(special_card=MAGICIAN_NAME),
                card_position=0,
                player=RandomPlayer(identifier=1),
            ),
            PlayedCard(
                card=Card(color=BASE_COLORS[2], number=13),
                card_position=2,
                player=player,
            ),
        ]
        results = GetTargetsForProbabilityEstimation(player, playable_cards_per_round, winner_per_round).execute()
        # TODO: Assertion
