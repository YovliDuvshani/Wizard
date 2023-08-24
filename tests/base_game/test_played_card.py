import pytest

from config.common import BASE_COLORS, JESTER_NAME, MAGICIAN_NAME, TRUMP_COLOR
from wizard.base_game.card import Card
from wizard.base_game.played_card import PlayedCard


class TestPlayedCard:
    @pytest.mark.parametrize(
        "first_card, second_card, expected_result",
        [
            pytest.param(
                PlayedCard(
                    card=Card(special_card=MAGICIAN_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(special_card=MAGICIAN_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                True,
                id="two_magicians",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(number=5, color=BASE_COLORS[1]),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(special_card=MAGICIAN_NAME),
                    starting_color=TRUMP_COLOR,
                    card_position=1,
                ),
                False,
                id="trump_vs_first_color",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(color=BASE_COLORS[2], number=6),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(color=BASE_COLORS[2], number=8),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                False,
                id="number_comparison",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(special_card=JESTER_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(special_card=JESTER_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                True,
                id="two_jesters",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(special_card=MAGICIAN_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(number=13, color=BASE_COLORS[1]),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                True,
                id="magician_vs_other",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(number=1, color=BASE_COLORS[1]),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                PlayedCard(
                    card=Card(number=13, color=BASE_COLORS[2]),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                True,
                id="first_color_vs_other_color",
            ),
            pytest.param(
                PlayedCard(
                    card=Card(special_card=JESTER_NAME),
                    starting_color=BASE_COLORS[1],
                    card_position=1,
                ),
                PlayedCard(
                    card=Card(number=13, color=BASE_COLORS[2]),
                    starting_color=BASE_COLORS[1],
                    card_position=0,
                ),
                False,
                id="jester_vs_other",
            ),
        ],
    )
    def test_gt_operator_returns_is_correct(
        self, first_card: PlayedCard, second_card: PlayedCard, expected_result: bool
    ):
        assert (first_card > second_card) == expected_result
