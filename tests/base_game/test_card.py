from config.common import BASE_COLORS, JESTER_NAME, MAGICIAN_NAME, TRUMP_COLOR
from wizard.base_game.card import Card


class TestCard:
    def test_eq_operator_returns_correct_result(self):
        assert Card(number=13, color=TRUMP_COLOR) == Card(number=13, color=TRUMP_COLOR)
        assert Card(special_card=MAGICIAN_NAME) == Card(special_card=MAGICIAN_NAME)
        assert Card(special_card=MAGICIAN_NAME) != Card(special_card=JESTER_NAME)

    def test_gt_operator_returns_correct_result(self):
        assert Card(number=1, color=TRUMP_COLOR) > Card(number=13, color=BASE_COLORS[1])
        assert Card(number=13, color=BASE_COLORS[1]) > Card(number=12, color=BASE_COLORS[1])
        assert Card(special_card=MAGICIAN_NAME) > Card(number=13, color=TRUMP_COLOR)
        assert Card(number=1, color=BASE_COLORS[3]) > Card(special_card=JESTER_NAME)
        assert Card(special_card=MAGICIAN_NAME) > Card(special_card=JESTER_NAME)

    def test_from_representation_outputs_right_card(self):
        assert Card.from_representation(Card(special_card=MAGICIAN_NAME).representation) == Card(
            special_card=MAGICIAN_NAME
        )
        assert Card.from_representation(Card(number=13, color=BASE_COLORS[1]).representation) == Card(
            number=13, color=BASE_COLORS[1]
        )
