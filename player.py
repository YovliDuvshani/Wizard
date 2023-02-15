from random import randint
from typing import List

import numpy as np

from adapters import ReadableStatTable
from card import Card


class Player:
    def __init__(self, id: str):
        self.cards = []
        self.initial_cards = []
        self.game = None
        self.id = id

    def assign_game(self, game):
        self.game = game

    def receive_cards(self, assigned_cards: List[Card]):
        self.cards = assigned_cards
        self.initial_cards = assigned_cards.copy()

    def _filter_playable_cards_relatively_to_color(self, color: str):
        cards_from_specified_color = []
        special_cards = []
        for card in self.cards:
            if card.color == color:
                cards_from_specified_color += [card]
            elif card.color is None:
                special_cards += [card]
        if cards_from_specified_color:
            return cards_from_specified_color + special_cards
        else:
            return self.cards

    def _playable_cards(self):
        # Retrieve the color that has been played
        for ind_card_played in range(len(self.game.current_turn_history)):
            first_color_played = self.game.current_turn_history[ind_card_played][1].color
            if first_color_played is not None:
                return self._filter_playable_cards_relatively_to_color(first_color_played)
        return self.cards

    def _possible_predictions(self):
        if len(self.game.initial_predictions) == (len(self.game.players) - 1):
            sum_of_already_announced_predictions = sum(
                [self.game.initial_predictions[player] for player in self.game.initial_predictions.keys()])
            forbidden_prediction = max(self.game.number_of_cards_per_player - sum_of_already_announced_predictions, 0)
            return list(range(forbidden_prediction)) + list(range(forbidden_prediction + 1, len(self.game.players) + 1))
        return list(range(len(self.game.players) + 1))

    def make_prediction(self):
        pass

    def play_card(self):
        pass

    def get_id(self):
        return self.id


class RandomPlayer(Player):
    def make_prediction(self):
        possible_predictions = self._possible_predictions()
        return np.random.choice(possible_predictions)

    def play_card(self):
        playable_cards = self._playable_cards()
        card_to_play = playable_cards[randint(0, len(playable_cards) - 1)]
        self.cards.remove(card_to_play)
        return card_to_play


class DeterministicPlayer(Player):
    def __init__(self, id: str, priority_cards_to_play: list[Card], prediction: int):
        super().__init__(id)
        self.priority_cards_to_play = priority_cards_to_play
        self.prediction = prediction

    def play_card(self):
        playable_cards = self._playable_cards()
        priority_card_was_played = False
        for card in self.priority_cards_to_play:
            if card in playable_cards:
                card_to_play = card
                priority_card_was_played = True

        if not priority_card_was_played:
            card_to_play = playable_cards[randint(0, len(playable_cards) - 1)]

        self.cards.remove(card_to_play)
        return card_to_play

    def make_prediction(self):
        possible_predictions = self._possible_predictions()
        prediction = self.prediction
        if prediction in possible_predictions:
            return prediction
        else:
            # if the prediction is not acceptable, return either prediction+1 or prediction-1 so that the prediction stays in the acceptable range.
            if len(self.game.number_of_cards_per_player) != prediction:
                return prediction + 1
            else:
                return prediction - 1


class StatisticPlayer(Player):
    def __init__(self, id: str, readable_stat_table: ReadableStatTable, use_base_strategy=True):
        super().__init__(id)
        self.readable_stat_table = readable_stat_table
        self.use_base_strategy = use_base_strategy

    def make_prediction(self, column_best_prediction: str = "best_prediction"):
        stats_for_hand = self.readable_stat_table.get_info_hand(self.cards)
        return stats_for_hand[column_best_prediction]

    def play_card(self, column_priority_plays= "Priority_plays"):
        if self.use_base_strategy:
            stats_for_hand = self.readable_stat_table.get_info_hand(self.initial_cards)
            priority_cards_to_play = stats_for_hand[column_priority_plays]
            playable_cards = self._playable_cards()
            for card in priority_cards_to_play:
                if card.is_in_list(playable_cards):
                    card_to_play = card
                    break

        else:  # Not implemented yet
            pass

        self.cards.remove(card_to_play)
        return card_to_play

class RealPlayer(Player):
    def __init__(self, id: str):
        super().__init__(id)

    def make_prediction(self):
        # Print Blabla
        prediction = input("Make a prediction: ")
        return prediction

    def play_card(self):
        # Print Blabla
        playable_cards = self._playable_cards()
        card_indices_to_play = input("Input the indices of the card you want to play among the playable cards: ")
        return playable_cards[card_indices_to_play]
