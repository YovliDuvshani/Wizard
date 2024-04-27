from typing import List

import pytest

from config.common import BASE_COLORS, JESTER_NAME, TRUMP_COLOR
from wizard.base_game.card import Card
from wizard.base_game.deck import Deck
from wizard.exhaustive_simulation.hand_combinations import HandCombinationsTwoCards


class TestHandCombinationsTwoCards:
    @pytest.mark.parametrize(
        "list_cards, expected_combination",
        [
            pytest.param(
                [Card(color=TRUMP_COLOR, number=9), Card(color=TRUMP_COLOR, number=10)],
                [Card(color=TRUMP_COLOR, number=9), Card(color=TRUMP_COLOR, number=10)],
                id="two_trumps",
            ),
            pytest.param(
                [Card(color=TRUMP_COLOR, number=10), Card(special_card=JESTER_NAME)],
                [Card(color=TRUMP_COLOR, number=10), Card(special_card=JESTER_NAME)],
                id="trump_jester",
            ),
            pytest.param(
                [
                    Card(color=BASE_COLORS[3], number=10),
                    Card(color=BASE_COLORS[1], number=9),
                ],
                [
                    Card(color=BASE_COLORS[1], number=10),
                    Card(color=BASE_COLORS[2], number=9),
                ],
                id="two_different_base_colors_desc",
            ),
            pytest.param(
                [
                    Card(color=BASE_COLORS[1], number=9),
                    Card(color=BASE_COLORS[3], number=10),
                ],
                [
                    Card(color=BASE_COLORS[2], number=9),
                    Card(color=BASE_COLORS[1], number=10),
                ],
                id="two_different_base_colors_asc",
            ),
            pytest.param(
                [
                    Card(color=BASE_COLORS[2], number=9),
                    Card(color=BASE_COLORS[2], number=10),
                ],
                [
                    Card(color=BASE_COLORS[1], number=9),
                    Card(color=BASE_COLORS[1], number=10),
                ],
                id="two_same_base_colors",
            ),
            pytest.param(
                [
                    Card(color=TRUMP_COLOR, number=9),
                    Card(color=BASE_COLORS[2], number=10),
                ],
                [
                    Card(color=TRUMP_COLOR, number=9),
                    Card(color=BASE_COLORS[1], number=10),
                ],
                id="trump_and_base_color",
            ),
            pytest.param(
                [
                    Card(color=BASE_COLORS[2], number=10),
                    Card(color=TRUMP_COLOR, number=9),
                ],
                [
                    Card(color=BASE_COLORS[1], number=10),
                    Card(color=TRUMP_COLOR, number=9),
                ],
                id="base_color_and_trump",
            ),
        ],
    )
    def test_list_cards_to_hand_combination_return_is_correct(
        self, list_cards: List[Card], expected_combination: List[Card]
    ):
        assert (
            HandCombinationsTwoCards(Deck()).list_cards_to_hand_combination(list_cards)
            == expected_combination
        )
