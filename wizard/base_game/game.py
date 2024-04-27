# mypy: disable-error-code="union-attr"
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.common import NUMBER_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS, TRUMP_COLOR
from wizard.base_game.card import Card
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.played_card import PlayedCard
from wizard.base_game.player import Player


@dataclass
class GameDefinition:
    initial_player_starting: Player
    trump_card_removed: Card
    players: List[Player]
    deck: Deck


@dataclass
class GameState:
    predictions: Dict[Player, Optional[int]]
    number_of_turns_won: Dict[Player, int]
    player_starting: Player
    current_turn_history: List[PlayedCard]
    previous_turns_history: List[List[PlayedCard]]
    starting_color: Optional[str] = None


class Game:
    def __init__(self, id_game: Optional[int] = None):
        self.id_game = id_game
        self.definition: Optional[GameDefinition] = None
        self.state: Optional[GameState] = None

    def initialize_game(
        self, deck: Deck, players: List[Player], first_player: Player
    ) -> None:
        self._assign_players(players)
        trump_card_removed = self._remove_one_trump_card(deck)
        self._distribute_cards(players=players, deck=deck)

        self.definition = GameDefinition(
            initial_player_starting=first_player,
            trump_card_removed=trump_card_removed,
            players=players,
            deck=deck,
        )
        self._initialize_game_state()

    def _assign_players(self, players: List[Player]) -> None:
        for player in players:
            player.assign_game(self)

    @staticmethod
    def _remove_one_trump_card(deck: Deck) -> Card:  # type: ignore
        for index, card in enumerate(deck.cards):
            if card.color == TRUMP_COLOR:
                deck.cards = deck.cards[:index] + deck.cards[index + 1 :]
                return card

    @staticmethod
    def _distribute_cards(players: List[Player], deck: Deck) -> None:
        assert players is not None, "No players"
        assert deck is not None, "Deck of cards is missing"
        assert NUMBER_CARDS_PER_PLAYER * NUMBER_OF_PLAYERS < len(
            deck.cards
        ), "Not enough cards"

        for player in players:
            player_has_received_cards = player.receive_cards(
                deck.cards[0:NUMBER_CARDS_PER_PLAYER]
            )
            if player_has_received_cards:
                deck.cards = deck.cards[NUMBER_CARDS_PER_PLAYER:]

    def _initialize_game_state(self) -> None:
        self.state = GameState(
            predictions=dict.fromkeys(self.definition.players, None),
            number_of_turns_won=dict.fromkeys(self.definition.players, 0),
            player_starting=self.definition.initial_player_starting,
            current_turn_history=[],
            previous_turns_history=[],
        )

    def request_predictions(self) -> None:
        for player in self.ordered_list_players:
            self.state.predictions[player] = player.make_prediction()

    def play_round(self, print_results: bool = False) -> None:
        for _ in range(NUMBER_CARDS_PER_PLAYER):
            self._play_turn(print_results=print_results)

    def _play_turn(self, print_results: bool) -> None:
        for position, player in enumerate(self.ordered_list_players):
            card = player.play_card()
            if card.color and not self.state.starting_color:
                self.state.starting_color = card.color
            self.state.current_turn_history += [
                PlayedCard(
                    card=card,
                    starting_color=self.state.starting_color,
                    card_position=position,
                    player=player,
                )
            ]

        winner = max(self.state.current_turn_history)
        self.state.number_of_turns_won[winner.player] += 1
        self.state.player_starting = winner.player
        self.state.previous_turns_history += [self.state.current_turn_history]
        self.state.current_turn_history = []
        self.state.starting_color = None
        if print_results:
            GameDisplayer(self).display_result_turn(
                played_cards=self.state.current_turn_history, winner=winner
            )

    @property
    def ordered_list_players(self) -> List[Player]:
        index_starting_player = self.definition.players.index(
            self.state.player_starting
        )
        return (
            self.definition.players[index_starting_player:]
            + self.definition.players[:index_starting_player]
        )

    def reset_game(self) -> None:
        self._initialize_game_state()
        for player in self.definition.players:
            player.reset_hand()


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
            print(
                f"Player {player.identifier} declared {self.game.state.predictions[player]}"
            )

    def display_result_each_player(self):
        print("--- RESULT EACH PLAYER ---")
        points_each_player = CountPoints().count_points_round(
            predictions=self.game.state.predictions,
            number_of_turns_won=self.game.state.number_of_turns_won,
        )
        for player in self.game.definition.players:
            print(
                f"Player {player.identifier} annonced {self.game.state.predictions[player]}"
                f" won {self.game.state.number_of_turns_won[player]}"
                f" got {points_each_player[player.identifier]} points"
            )

    def display_result_turn(self, played_cards: List[PlayedCard], winner: PlayedCard):
        print(f"TURN {len(self.game.state.previous_turns_history)}", end=" ")
        played_cards.sort(key=lambda played_card: played_card.player.identifier)
        for played_card in played_cards:
            outcome = "WINNER" if winner == played_card else "LOSER"
            print(
                f"--- {outcome}: player {played_card.player.identifier}"
                f" with {played_card.card.representation}",
                end=" ",
            )
        print()
