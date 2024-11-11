# mypy: disable-error-code="attr-defined"
import random
from functools import partial
from typing import List, Optional, Type

import pandas as pd

from config.common import BASE_COLORS
from wizard.base_game.card import Card
from wizard.base_game.deck import Deck
from wizard.base_game.player.card_play_policy import (
    BaseCardPlayPolicy,
    DefinedCardPlayPolicy,
    DQNCardPlayPolicy,
    HighestCardPlayPolicy,
    RandomCardPlayPolicy,
    StatisticalCardPlayPolicy,
)
from wizard.base_game.player.prediction_policy import (
    BasePredictionPolicy,
    DefinedPredictionPolicy,
    DQNPredictionPolicy,
    RandomPredictionPolicy,
    StatisticalPredictionPolicy,
)
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent


class Player:
    def __init__(
        self,
        identifier: int,
        prediction_policy: Type[BasePredictionPolicy],
        card_play_policy: Type[BaseCardPlayPolicy],
        name: str | None = None,
        set_prediction: int | None = None,
        set_card_play_priority: list[Card] | None = None,
        stat_table: pd.DataFrame | None = None,
        agent: DQNAgent | None = None,
    ):
        self.identifier = identifier
        self.name = name
        self.card_play_policy = card_play_policy
        self.prediction_policy = prediction_policy

        self.cards: Optional[List[Card]] = None
        self.game = None
        self.initial_cards: Optional[List[Card]] = None
        self.colors_known_to_not_be_in_hand = []

        self.set_prediction = set_prediction
        self.set_card_play_priority = set_card_play_priority
        self.stat_table = stat_table
        self.agent = agent

    def assign_game(self, game):
        self.game = game

    def receive_cards(self, cards: List[Card]) -> bool:
        if not self.cards:
            self.cards = cards
            self.initial_cards = cards.copy()
            self.colors_known_to_not_be_in_hand = []
            return True
        return False

    def drop_hand(self):
        self.cards = []
        self.colors_known_to_not_be_in_hand = []

    def reset_hand(self) -> None:
        self.cards = self.initial_cards.copy()
        self.colors_known_to_not_be_in_hand = []

    def make_prediction(self) -> int:
        return self.prediction_policy(self).execute()

    def play_card(self, remove_card: bool = True, card_to_play: Card | None = None) -> Card:
        if not card_to_play:
            card_to_play = self.select_card_to_play()
        if remove_card:
            for card in self.cards:
                if card == card_to_play:
                    self.cards.remove(card)
                    break
        self._update_colors_known_to_not_be_in_hand(card_to_play)
        return card_to_play

    def select_card_to_play(self) -> Card:
        return self.card_play_policy(self).execute()

    def provide_strategy(
        self, set_prediction: int | None = None, set_card_play_priority: list[Card] | None = None
    ) -> None:
        self.set_prediction = set_prediction
        self.set_card_play_priority = set_card_play_priority

    @property
    def position(self) -> int:
        return self.game.ordered_list_players.index(self)

    def sample_another_possible_hand(self):
        other_players_cards = [
            card for player in self.game.definition.players if player != self for card in player.cards
        ]
        trump_card_removed = self.game.definition.trump_card_removed
        possible_cards = Deck()
        possible_cards.remove_cards(other_players_cards + [trump_card_removed])
        possible_cards = possible_cards.filter_deck(
            colors=list(set(BASE_COLORS) - set(self.colors_known_to_not_be_in_hand))
        )
        self.cards = random.sample(possible_cards, len(self.cards))
        self.initial_cards = (self._cards_already_played + self.cards).copy()

    @property
    def _cards_already_played(self):
        return list(set(self.initial_cards) - set(self.cards))

    def _update_colors_known_to_not_be_in_hand(self, played_card: Card):
        starting_color = self.game.state.round_specifics.starting_color
        if (played_card.color and starting_color) and played_card.color != starting_color:
            self.colors_known_to_not_be_in_hand.append(starting_color)


RandomPlayer = partial(
    Player,
    prediction_policy=RandomPredictionPolicy,
    card_play_policy=RandomCardPlayPolicy,
)
MaxRandomPlayer = partial(
    Player,
    prediction_policy=RandomPredictionPolicy,
    card_play_policy=HighestCardPlayPolicy,
)
DefinedStrategyPlayer = partial(
    Player,
    prediction_policy=DefinedPredictionPolicy,
    card_play_policy=DefinedCardPlayPolicy,
)
StatisticalPlayer = partial(
    Player,
    prediction_policy=StatisticalPredictionPolicy,
    card_play_policy=StatisticalCardPlayPolicy,
)
DQNPlayer = partial(Player, prediction_policy=DQNPredictionPolicy, card_play_policy=DQNCardPlayPolicy)
