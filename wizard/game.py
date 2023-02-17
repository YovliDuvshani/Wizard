from typing import List, Dict, Optional

from config.common import (
    NUMBER_OF_PLAYERS,
    NUMBER_CARDS_PER_PLAYER,
    TRUMP_COLOR,
    DYNAMIC_REWARD,
    BASE_REWARD,
    DYNAMIC_LOSS,
)
from wizard.card import Card
from wizard.deck import Deck
from wizard.player import Player
from wizard.evaluate_turn import EvaluateTurn
from wizard.played_card import PlayedCard


class Game:
    def __init__(self, id_game: int):
        self.id_game = id_game
        self.current_turn_history: List[PlayedCard] = []
        self.previous_turns_history: List[List[PlayedCard]] = []
        self.initial_predictions: Dict[Player, int] = {}
        self.number_of_turns_won: Dict[Player, int] = {}
        self.players: Optional[List[Player]] = None
        self.deck: Optional[Deck] = None
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

    def _assign_deck(self, deck: Deck) -> None:
        self.deck = deck

    def _assign_players(self, players: List[Player]) -> None:
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
            player.cards = self.deck.cards[0:NUMBER_CARDS_PER_PLAYER]
            self.deck.cards = self.deck.cards[NUMBER_CARDS_PER_PLAYER:]

    def request_predictions(self) -> None:
        for player in self.players:
            self.initial_predictions[player] = player.make_prediction()

    def play_turn(self, print_results: bool) -> None:
        starting_color = None
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
        self.current_turn_history = []

    def play_round(self, print_results: bool = False) -> None:
        for turn in range(NUMBER_CARDS_PER_PLAYER):
            self.play_turn(print_results=print_results)

    @property
    def ordered_list_players(self) -> List[Player]:
        index_starting_player = self.players.index(self.player_starting)
        return (
            self.players[index_starting_player:] + self.players[:index_starting_player]
        )

    def count_points(self):
        score: Dict[Player, int] = {}
        for player in self.players:
            if self.initial_predictions[player] == self.number_of_turns_won[player]:
                score[player] = int(
                    (DYNAMIC_REWARD * self.initial_predictions[player] + BASE_REWARD)
                )
            else:
                score[player] = int(
                    DYNAMIC_LOSS
                    * abs(
                        self.initial_predictions[player]
                        - self.number_of_turns_won[player]
                    )
                )
        return score

    @property
    def number_played_turns(self) -> int:
        return len(self.previous_turns_history)

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
        for player in self.players:
            print(
                f"Player {player.identifier} annonced {self.initial_predictions[player]}"
                f" won {self.number_of_turns_won[player]} got {self.count_points()[player]} points"
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
