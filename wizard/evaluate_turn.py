from typing import List

from wizard.played_card import PlayedCard


class EvaluateTurn:
    def __init__(self, played_cards: List[PlayedCard]):
        self.played_cards = played_cards

    def evaluate_winner(self) -> PlayedCard:
        return max(self.played_cards)
