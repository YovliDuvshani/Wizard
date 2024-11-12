# mypy: disable-error-code="union-attr"
import random
from dataclasses import dataclass
from random import choice
from typing import Dict, List, Optional

from config.common import NUMBER_OF_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS, TRUMP_COLOR
from wizard.base_game.card import Card
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.played_card import PlayedCard
from wizard.base_game.player.player import Player

Terminal = bool


@dataclass
class GameDefinition:
    initial_starting_player: Player
    trump_card_removed: Card
    players: List[Player]
    deck: Deck


@dataclass
class GameRoundSpecifics:
    turn_history: List[PlayedCard]
    starting_player: Player
    number_cards_played: int
    starting_color: Optional[str] = None


@dataclass
class GameState:
    predictions: Dict[Player, Optional[int]]
    number_of_turns_won: Dict[Player, int]
    winner_history: List[PlayedCard]
    previous_turns_history: List[List[PlayedCard]]
    round_specifics: GameRoundSpecifics


class Game:
    def __init__(self, id_game: Optional[int] = None):
        self.id_game = id_game
        self.definition: Optional[GameDefinition] = None
        self.state: Optional[GameState] = None

    def initialize_game(
        self, deck: Deck, players: List[Player], starting_player: Player | None = None, deterministic: bool = False
    ) -> None:
        self._assign_players(players)
        trump_card_removed = self._remove_one_trump_card(deck, deterministic)

        self._distribute_cards(players=players, deck=deck)

        starting_player = starting_player if starting_player else random.choice(players)

        self.definition = GameDefinition(
            initial_starting_player=starting_player,
            trump_card_removed=trump_card_removed,
            players=players,
            deck=deck,
        )
        self._initialize_game_state()

    def request_predictions(self) -> None:
        for player in self.ordered_list_players:
            self.state.predictions[player] = player.make_prediction()

    def play_game(self, print_results: bool = False) -> None:
        terminal = self._play_next_card(print_results=print_results)
        while not terminal:
            terminal = self._play_next_card(print_results=print_results)

    def reset_game(self) -> None:
        self._initialize_game_state()
        for player in self.definition.players:
            player.reset_hand()

    def get_to_prediction_state_for_given_player(self, player: Player):
        for next_player_predicting in self.ordered_list_players:
            if next_player_predicting is player:
                break
            self.set_prediction_for_given_player(next_player_predicting)

    def set_prediction_for_given_player(self, player: Player, prediction: int | None = None):
        self.state.predictions[player] = prediction if prediction is not None else player.make_prediction()

    def get_to_first_play_afterstate_for_given_player(self, player: Player) -> Terminal:
        for next_player_predicting in self.ordered_list_players[self.ordered_list_players.index(player) + 1 :]:
            self.state.predictions[next_player_predicting] = next_player_predicting.make_prediction()
        while self.next_player_playing != player:
            self._play_next_card(print_results=False)
        return False

    def get_to_next_play_afterstate_for_given_player(
        self, player: Player, action: str | None = None, print_results: bool = False
    ) -> Terminal:
        assert self.next_player_playing == player, "Learning player should have been playing"

        if action:
            action = Card.from_representation(action)

        terminal = self._play_next_card(action, print_results)
        while (self.next_player_playing != player) & (not terminal):
            terminal = self._play_next_card(print_results=print_results)
        return terminal

    def _assign_players(self, players: List[Player]) -> None:
        for player in players:
            player.assign_game(self)

    @staticmethod
    def _remove_one_trump_card(deck: Deck, deterministic: bool) -> Card:  # type: ignore
        remaining_trump_cards = [card for card in deck.cards if card.color == TRUMP_COLOR]
        trump_card_to_remove = remaining_trump_cards[-1] if deterministic else choice(remaining_trump_cards)
        index_trump_card_to_remove = deck.cards.index(trump_card_to_remove)
        deck.cards = deck.cards[:index_trump_card_to_remove] + deck.cards[index_trump_card_to_remove + 1 :]
        return trump_card_to_remove

    @staticmethod
    def _distribute_cards(players: List[Player], deck: Deck) -> None:
        assert players is not None, "No players"
        assert deck is not None, "Deck of cards is missing"
        assert NUMBER_OF_CARDS_PER_PLAYER * NUMBER_OF_PLAYERS < len(deck.cards), "Not enough cards"

        for player in players:
            player_has_received_cards = player.receive_cards(deck.cards[0:NUMBER_OF_CARDS_PER_PLAYER])
            if player_has_received_cards:
                deck.cards = deck.cards[NUMBER_OF_CARDS_PER_PLAYER:]

    def _initialize_game_state(self) -> None:
        self.state = GameState(
            predictions=dict.fromkeys(self.definition.players, None),
            number_of_turns_won=dict.fromkeys(self.definition.players, 0),
            previous_turns_history=[],
            winner_history=[],
            round_specifics=GameRoundSpecifics(
                turn_history=[],
                starting_player=self.definition.initial_starting_player,
                number_cards_played=0,
                starting_color=None,
            ),
        )

    def _play_next_card(self, card: Card | None = None, print_results: bool = False) -> Terminal:
        player = self.next_player_playing
        card = player.play_card(card_to_play=card)
        if card.color and not self.state.round_specifics.starting_color:
            self.state.round_specifics.starting_color = card.color
        self.state.round_specifics.turn_history += [
            PlayedCard(
                card=card,
                starting_color=self.state.round_specifics.starting_color,
                card_position=self.state.round_specifics.number_cards_played,
                player=player,
            )
        ]
        self.state.round_specifics.number_cards_played += 1

        if len(self.state.round_specifics.turn_history) == len(self.definition.players):
            self._complete_round(print_results)
            if len(self.state.previous_turns_history) == NUMBER_OF_CARDS_PER_PLAYER:
                return True
        return False

    def _complete_round(self, print_results: bool):
        winner = max(self.state.round_specifics.turn_history)

        self.state.number_of_turns_won[winner.player] += 1
        self.state.winner_history += [winner]
        self.state.previous_turns_history += [self.state.round_specifics.turn_history]
        if print_results:
            GameDisplayer(self).display_result_round(
                played_cards=self.state.round_specifics.turn_history, winner=winner
            )

        self.state.round_specifics.starting_player = winner.player
        self.state.round_specifics.turn_history = []
        self.state.round_specifics.starting_color = None
        self.state.round_specifics.number_cards_played = 0

    @property
    def ordered_list_players(self) -> List[Player]:
        index_starting_player = self.definition.players.index(self.state.round_specifics.starting_player)
        return self.definition.players[index_starting_player:] + self.definition.players[:index_starting_player]

    @property
    def initial_ordered_list_players(self) -> List[Player]:
        index_initial_starting_player = self.definition.players.index(self.definition.initial_starting_player)
        return (
            self.definition.players[index_initial_starting_player:]
            + self.definition.players[:index_initial_starting_player]
        )

    @property
    def next_player_playing(self):
        return self.ordered_list_players[self.state.round_specifics.number_cards_played]


class GameDisplayer:
    def __init__(self, game: Game):
        self.game = game

    def display_hand_each_player(self):
        print("--- HAND EACH PLAYER ---")
        for player in self.game.definition.players:
            print(f"Player {player.identifier} hand", end=" ")
            for card in player.cards:
                print(f"- {card.representation} ", end="")
            print()

    def display_prediction_each_player(self):
        print("--- PREDICTION EACH PLAYER ---")
        for player in self.game.definition.players:
            print(f"Player {player.identifier} declared {self.game.state.predictions[player]}")

    def display_result_each_player(self):
        print("--- RESULT EACH PLAYER ---")
        points_each_player = CountPoints().execute(
            predictions=self.game.state.predictions,
            number_of_turns_won=self.game.state.number_of_turns_won,
        )
        for player in self.game.definition.players:
            print(
                f"Player {player.identifier} annonced {self.game.state.predictions[player]}"
                f" won {self.game.state.number_of_turns_won[player]}"
                f" got {points_each_player[player.identifier]} points"
            )

    def display_result_round(self, played_cards: List[PlayedCard], winner: PlayedCard):
        print(f"TURN {len(self.game.state.previous_turns_history)}", end=" ")
        played_cards.sort(key=lambda played_card: played_card.player.identifier)
        for played_card in played_cards:
            outcome = "WINNER" if winner == played_card else "LOSER"
            print(
                f"--- {outcome}: player {played_card.player.identifier}" f" with {played_card.card.representation}",
                end=" ",
            )
        print()
