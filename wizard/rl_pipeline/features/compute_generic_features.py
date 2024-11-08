from functools import cached_property
from typing import List

from config.common import (
    BASE_COLORS,
    JESTER_NAME,
    MAGICIAN_NAME,
    NUMBER_OF_CARDS_PER_PLAYER,
    TRUMP_COLOR,
)
from wizard.base_game.card import Card
from wizard.base_game.game import Game
from wizard.base_game.played_card import PlayedCard
from wizard.base_game.player.player import Player
from wizard.rl_pipeline.features.data_cls import (
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericFeatures,
    GenericObjectiveContextFeatures,
)


class ComputeGenericFeatures:
    def __init__(self, game: Game, player: Player):
        self._game = game
        self._player = player

    def execute(
        self,
    ) -> GenericFeatures:
        card_specific_features = {
            card.representation: self._compute_card_specific_features(card) for card in self._player.cards
        }  # Duplicates are removed implicitly
        card_context_features = self._compute_card_context_features()
        objective_context_features = self._compute_objective_context_features()
        return GenericFeatures(card_specific_features, card_context_features, objective_context_features)

    def _compute_card_specific_features(self, card: Card) -> GenericCardSpecificFeatures:
        kwargs = {}
        kwargs.update(self._compute_base_card_feature(card))
        kwargs.update(self._compute_outcome_of_playing_card_feature(card))
        kwargs.update(self._compute_advanced_card_feature(card))
        return GenericCardSpecificFeatures(**kwargs)

    def _compute_base_card_feature(self, card: Card):
        return {
            "IS_PLAYABLE": card in self._player.card_play_policy(self._player).playable_cards(),
            "IS_TRUMP": card.color == TRUMP_COLOR,
            "IS_MAGICIAN": card.special_card == MAGICIAN_NAME,
            "IS_JESTER": card.special_card == JESTER_NAME,
            "COLOR": (BASE_COLORS.index(card.color) + 1) / 4 if card.color else 0,  # TODO: Workaround
            "NUMBER": card.number / 13 if card.number else 0,  # TODO: Workaround
        }

    def _compute_advanced_card_feature(self, card: Card):
        cards_remaining_same_color = [c for c in self.remaining_cards if c.color == card.color]
        return {
            **self._compute_card_statistics_on_subset(
                card=card,
                suffix="REMAINING_SAME_COLOR",
                cards_subset=cards_remaining_same_color,
            ),
            **self._compute_card_statistics_on_subset(
                card=card,
                suffix="SAME_COLOR_IN_PLAYERS_HAND",
                cards_subset=[c for c in self._player.cards if c.color == card.color],
            ),
            **self._compute_card_statistics_on_subset(
                card=card,
                suffix="REMAINING_AMONG_SPECIAL_TRUMP_AND_SAME_COLOR",
                cards_subset=cards_remaining_same_color + self.remaining_trump_cards + self.remaining_special_cards,
            ),
        }

    @staticmethod
    def _compute_card_statistics_on_subset(card: Card, suffix: str, cards_subset: List[Card]):
        return {
            f"NUMBER_CARDS_{suffix}": len(cards_subset),
            f"NUMBER_SUPERIOR_CARDS_{suffix}": len([c for c in cards_subset if c > card]),
            f"NUMBER_INFERIOR_CARDS_{suffix}": len([c for c in cards_subset if c > card]),
        }

    def _compute_outcome_of_playing_card_feature(self, card: Card):
        kwargs = {}
        current_played_cards = self._game.state.round_specifics.turn_history
        hypothetically_played_card = PlayedCard(
            card=card,
            card_position=len(current_played_cards),
            player=self._player,
            starting_color=self._game.state.round_specifics.starting_color,
        )
        if max([hypothetically_played_card] + current_played_cards) is hypothetically_played_card:
            kwargs["CAN_WIN_CURRENT_SUB_ROUND"] = True
            if (
                len(current_played_cards) == len(self._game.definition.players) - 1
                or card.special_card == MAGICIAN_NAME
            ):
                kwargs["WILL_WIN_CURRENT_SUB_ROUND"] = True
        return kwargs

    def _compute_card_context_features(self) -> GenericCardsContextFeatures:
        return GenericCardsContextFeatures(
            NUMBER_CARDS_REMAINING=len(self.remaining_cards),
            NUMBER_CARDS_REMAINING_IN_OTHER_PLAYERS_HANDS=len(self.remaining_cards_in_other_players_hand),
            NUMBER_CARDS_REMAINING_IN_PLAYER_HAND=len(self._player.cards),
            NUMBER_CARDS_REMAINING_PER_COLOR={
                color: len([card for card in self.remaining_cards if card.color == color]) for color in BASE_COLORS
            },
            NUMBER_CARDS_REMAINING_IN_HAND_PER_COLOR={
                color: len([card for card in self._player.cards if card.color == color]) for color in BASE_COLORS
            },
        )

    def _compute_objective_context_features(self) -> GenericObjectiveContextFeatures:
        return GenericObjectiveContextFeatures(
            NUMBER_ROUNDS_TO_WIN=(
                self._game.state.predictions[self._player]
                if self._game.state.predictions[self._player] is not None
                else 0
            ),
            NUMBER_ROUNDS_ALREADY_WON=(
                self._game.state.number_of_turns_won[self._player]
                if self._game.state.predictions[self._player] is not None
                else 0
            ),
            TOTAL_NUMBER_OF_ROUNDS=NUMBER_OF_CARDS_PER_PLAYER,
            IS_PLAYER_STARTING=self._game.next_player_playing is self._player,
            PLAYER_POSITION=self._game.ordered_list_players.index(self._player),
            IS_TERMINAL=not self._player.cards,
        )

    @cached_property
    def remaining_cards_in_other_players_hand(self):
        return [card for player in self._game.definition.players if player is not self._player for card in player.cards]

    @cached_property
    def remaining_cards(self):
        remaining_cards_in_deck = self._game.definition.deck.cards
        return remaining_cards_in_deck + self.remaining_cards_in_other_players_hand

    @cached_property
    def remaining_trump_cards(self):
        return [c for c in self.remaining_cards if c.color == TRUMP_COLOR]

    @cached_property
    def remaining_special_cards(self):
        return [c for c in self.remaining_cards if c.special_card is not None]
