from dataclasses import dataclass
from typing import List, Any

from wizard.base_game.card import Card
from wizard.base_game.played_card import PlayedCard
from wizard.base_game.player import Player


@dataclass
class ProbabilityTargets:
    card: Card
    prob_win_after: float
    prob_win_time_t: float


class GetTargetsForProbabilityEstimation:
    def __init__(
        self,
        player: Player,
        playable_cards_per_round: List[List[Card]],
        winner_per_round: List[PlayedCard],
    ):
        self._player = player
        self._playable_cards_per_round = playable_cards_per_round
        self._winner_per_round = winner_per_round

    def execute(self):
        return [
            ProbabilityTargets(
                card=card,
                prob_win_after=self._get_target_prob_win_after(card),
                prob_win_time_t=self._get_target_prob_win_time_t(card),
            )
            for card in self.remove_duplicates(self._flattened_playable_cards)
        ]

    def _get_target_prob_win_time_t(self, card: Card):
        return (
            1
            if card
            in [w_played_card.card for w_played_card in self._winner_per_round if w_played_card.player == self._player]
            else 0
        )

    def _get_target_prob_win_after(self, card: Card):
        cards_not_directly_played = self.remove_duplicates(
            [
                c
                for c in self._flattened_playable_cards
                if self._flattened_playable_cards.count(c) >= 2
            ]
        )
        if card in cards_not_directly_played:
            return (
                1
                if card
                in [w_played_card.card for w_played_card in self._winner_per_round if w_played_card.player == self._player]
                else 0
            )
        return None

    @property
    def _flattened_playable_cards(self):
        return [
            playable_card
            for cards_per_round in self._playable_cards_per_round
            for playable_card in cards_per_round
        ]

    @staticmethod
    def remove_duplicates(l: List[Any]):
        new_list = []
        for ele in l:
            if ele not in new_list:
                new_list.append(ele)
        return new_list
