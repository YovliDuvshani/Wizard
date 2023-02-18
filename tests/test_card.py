from config.common import MAGICIAN_NAME, TRUMP_COLOR
from wizard.card import Card


class TestCard:
    def test_eq_operator_returns_correct_result(self):
        assert Card(number=13, color=TRUMP_COLOR) == Card(number=13, color=TRUMP_COLOR)
        assert Card(special_card=MAGICIAN_NAME) == Card(special_card=MAGICIAN_NAME)
