# mypy: disable-error-code="attr-defined"
from functools import partial
from typing import List, Optional, Type

import pandas as pd

from wizard.base_game.card import Card
from wizard.base_game.player.card_play_policy import (
    BaseCardPlayPolicy,
    DefinedCardPlayPolicy,
    DQNCardPlayPolicy,
    HighestCardPlayPolicy,
    LowestCardPlayPolicy,
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

        self.set_prediction = set_prediction
        self.set_card_play_priority = set_card_play_priority
        self.stat_table = stat_table
        self.agent = agent

    def assign_game(self, game):
        self.game = game

    def receive_cards(self, cards: List[Card]):
        if not self.cards:
            self.cards = cards
            self.initial_cards = cards.copy()
            return True
        return False

    def reset_hand(self) -> None:
        self.cards = self.initial_cards.copy()

    def make_prediction(self) -> int:
        return self.prediction_policy(self).execute()

    def play_card(self, remove_card: bool = True, card: Card | None = None) -> Card:
        if not card:
            card = self.select_card_to_play()
        if remove_card:
            self.cards.remove(card)
        return card

    def select_card_to_play(self) -> Card:
        return self.card_play_policy(self).execute()

    def provide_strategy(self, set_prediction: int, set_card_play_priority: list[Card] | None):
        self.set_prediction = set_prediction
        self.set_card_play_priority = set_card_play_priority


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
