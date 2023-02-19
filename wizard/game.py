from typing import List, Dict, Optional

from config.common import (
    NUMBER_OF_PLAYERS,
    NUMBER_CARDS_PER_PLAYER,
    TRUMP_COLOR,
)
from wizard.card import Card
from wizard.count_points import CountPoints
from wizard.deck import Deck
from wizard.player import Player
from wizard.evaluate_turn import EvaluateTurn
from wizard.played_card import PlayedCard


class Game:
    def __init__(self, id_game: Optional[int] = None):
        self.id_game = id_game
        self.current_turn_history: Optional[List[PlayedCard]] = None
        self.previous_turns_history: Optional[List[List[PlayedCard]]] = None
        self.initial_predictions: Optional[Dict[Player, int]] = None
        self.number_of_turns_won: Optional[Dict[Player, int]] = None
        self.players: Optional[List[Player]] = None
        self.deck: Optional[Deck] = None
        self.initial_player_starting: Optional[Player] = None
        self.player_starting: Optional[Player] = None
        self.trump_card_removed: Optional[Card] = None

    def initialize_game(
        self, deck: Deck, players: List[Player], first_player: Player
    ) -> None:
        self._assign_deck(deck)
        self._assign_players(players)
        self._remove_one_trump_card()
        self._distribute_cards()

        self.player_starting = first_player
        self.initial_player_starting = first_player

    def _assign_deck(self, deck: Deck) -> None:
        self.deck = deck

    def _assign_players(self, players: List[Player]) -> None:
        self.number_of_turns_won = {}
        self.players = players
        for player in self.players:
            player.assign_game(self)
            self.number_of_turns_won[player] = 0

    def _remove_one_trump_card(self) -> None:
        for index, card in enumerate(self.deck.cards):
            if card.color == TRUMP_COLOR:
                self.trump_card_removed = card
                self.deck.cards = (
                    self.deck.cards[:index] + self.deck.cards[(index + 1) :]
                )

    def _distribute_cards(self) -> None:
        assert self.players is not None, "No players"
        assert self.deck is not None, "Deck of cards is missing"
        assert NUMBER_CARDS_PER_PLAYER * NUMBER_OF_PLAYERS < len(
            self.deck.cards
        ), "Not enough cards"

        for player in self.players:
            player_has_received_cards = player.receive_cards(
                self.deck.cards[0:NUMBER_CARDS_PER_PLAYER]
            )
            if player_has_received_cards:
                self.deck.cards = self.deck.cards[NUMBER_CARDS_PER_PLAYER:]

    def request_predictions(self) -> None:
        self.initial_predictions = {}
        for player in self.players:
            self.initial_predictions[player] = player.make_prediction()

    def _play_turn(self, print_results: bool) -> None:
        self.current_turn_history = []
        starting_color: Optional[str] = None
        for position, player in enumerate(self.ordered_list_players):
            card = player.play_card()
            if card.color and (not starting_color):
                starting_color = card.color
            self.current_turn_history += [
                PlayedCard(
                    card=card,
                    starting_color=starting_color,
                    card_position=position,
                    player=player,
                )
            ]

        winner = EvaluateTurn(played_cards=self.current_turn_history).evaluate_winner()
        self.number_of_turns_won[winner.player] += 1
        self.player_starting = winner.player
        self.previous_turns_history += [self.current_turn_history]
        if print_results:
            self.display_result_turn(
                played_cards=self.current_turn_history, winner=winner
            )

    def play_round(self, print_results: bool = False) -> None:
        self.previous_turns_history = []
        for turn in range(NUMBER_CARDS_PER_PLAYER):
            self._play_turn(print_results=print_results)

    @property
    def ordered_list_players(self) -> List[Player]:
        index_starting_player = self.players.index(self.player_starting)
        return (
            self.players[index_starting_player:] + self.players[:index_starting_player]
        )

    def reset_game(self):
        self.player_starting = self.initial_player_starting
        for player in self.players:
            player.reset_hand()
            if self.number_of_turns_won:
                self.number_of_turns_won[player] = 0
            if self.initial_predictions:
                self.initial_predictions[player] = 0

    @property
    def number_played_turns(self) -> int:
        if self.previous_turns_history:
            return len(self.previous_turns_history)
        return 0

    def display_hand_each_player(self):
        print("--- HAND EACH PLAYER ---")
        for player in self.players:
            print(f"Player {player.identifier} hand", end=" ")
            for card in player.cards:
                print(f"- {card.representation} ", end="")
            print()

    def display_prediction_each_player(self):
        print("--- PREDICTION EACH PLAYER ---")
        for player in self.players:
            print(
                f"Player {player.identifier} declared {self.initial_predictions[player]}"
            )

    def display_result_each_player(self):
        print("--- RESULT EACH PLAYER ---")
        points_each_player = CountPoints().count_points_round(
            predictions=self.initial_predictions,
            number_of_turns_won=self.number_of_turns_won,
        )
        for player in self.players:
            print(
                f"Player {player.identifier} annonced {self.initial_predictions[player]}"
                f" won {self.number_of_turns_won[player]} got {points_each_player[player]} points"
            )

    def display_result_turn(self, played_cards: List[PlayedCard], winner: PlayedCard):
        print(f"TURN {self.number_played_turns}", end=" ")
        played_cards.sort(key=lambda played_card: played_card.player.identifier)
        for played_card in played_cards:
            outcome = "WINNER" if winner == played_card else "LOSER"
            print(
                f"--- {outcome}: player {played_card.player.identifier} with {played_card.card.representation}",
                end=" ",
            )
        print()
