from typing import List

from card import Card
from deck import Deck
from player import Player

NUMBER_OF_PLAYERS = 3
BASE_REWARD = 20
DYNAMIC_REWARD = 10
DYNAMIC_LOSS = - 10


class Game:
    def __init__(self, id_game: int, number_of_cards_per_player: int):
        self.number_of_cards_per_player = number_of_cards_per_player
        self.id_game = id_game
        self.current_turn_history = []
        self.play_history = []
        self.initial_predictions = {}
        self.number_of_turns_won = {}
        self.trump_card = None
        self.players = None
        self.deck = None
        self.player_starting = None

    def _assign_deck(self, deck: Deck):
        self.deck = deck

    def _assign_players(self, players: List[Player]):
        self.players = players
        for player in self.players:
            player.assign_game(self)

    def _distribute_cards(self):
        assert self.players is not None, 'No players'
        assert self.deck is not None, 'Deck of cards is missing'
        assert self.number_of_cards_per_player * NUMBER_OF_PLAYERS < len(self.deck.cards), 'Not enough cards'

        for player in self.players:
            player.cards = self.deck.cards[0:self.number_of_cards_per_player]
            self.deck.cards = self.deck.cards[self.number_of_cards_per_player:]

    def _reveal_trump_card(self):
        self.trump_card = self.deck.cards[0]
        self.deck.cards = self.deck.cards[1:]

    def start_game(self, deck: Deck, players: List[Player], first_player):

        self._assign_deck(deck)
        self._assign_players(players)
        self._distribute_cards()
        self._reveal_trump_card()
        self.player_starting = first_player

    def make_predictions(self):
        for player in self.players:
            self.initial_predictions[player] = player.make_prediction()
            self.number_of_turns_won[player] = 0

    def compare_two_players(self, card1: Card, card2: Card, player1: Player, player2: Player, starting_color: str):
        if (card1.special_card is None) & (card2.special_card is None):
            if [int(card1.color == self.trump_card.color), (int(card1.color == starting_color)), card1.number] > [int(card2.color == self.trump_card.color), (int(card2.color == starting_color)), card2.number]:
                return player1, card1
            return player2, card2
        elif (card1.special_card == 'Wizard') or (card2.special_card == 'Jester'):
            return player1, card1
        else:
            return player2, card2

    def _highest_player(self, turn_history: list[Player, Card]):
        best_player, best_card = turn_history[0]
        starting_color = best_card.color
        for player, card in turn_history[1:]:
            if starting_color is None:
                starting_color = card.color
            best_player, best_card = self.compare_two_players(best_card, card, best_player, player, starting_color)
        return best_player, best_card

    def play_turn(self, log):
        ordered_list_players = self._ordered_list_players()
        for player in ordered_list_players:
            card = player.play_card()
            self.current_turn_history += [[player, card]]

        winner, _ = self._highest_player(self.current_turn_history)
        self.number_of_turns_won[winner] += 1

        if log:
            player_to_player_id_ready_to_display = self.player_to_player_id_ready_to_display(self.current_turn_history)
            player_to_player_id_ready_to_display.sort()
            for player_id, card in player_to_player_id_ready_to_display:
                if player_id == winner.id:
                    print(f'--- WINNER: player {player_id} with {card.prepare_card_for_display()}', end=' ')
                else:
                    print(f'--- LOSER: player {player_id} with {card.prepare_card_for_display()}', end=' ')
            print()

        self.player_starting = winner

        self.play_history += [self.current_turn_history]
        self.current_turn_history = []

    def play_game(self, log):
        for turn in range(self.number_of_cards_per_player):
            self.play_turn(log)

    def _ordered_list_players(self):
        index_starting_player = self.players.index(self.player_starting)
        return self.players[index_starting_player:] + self.players[:index_starting_player]

    def count_points(self):
        score = {}
        for player in self.players:
            if self.initial_predictions[player] == self.number_of_turns_won[player]:
                score[player] = DYNAMIC_REWARD * self.initial_predictions[player] + BASE_REWARD
            else:
                score[player] = DYNAMIC_LOSS * abs(self.initial_predictions[player] - self.number_of_turns_won[player])
        return score

    @staticmethod
    def player_to_player_id_ready_to_display(list_player_cards: list[Player, object]):
        new_list_player = []
        for element in list_player_cards:
            new_list_player.append([element[0].id, element[1]])
        return new_list_player


class PredefinedGameForFirstPlayer(Game):
    def __init__(self, id_game: int, number_of_cards_per_player: int, cards_first_player: list[Card],
                 prediction_first_player: int):
        super().__init__(id_game, number_of_cards_per_player)
        self.prediction_first_player = prediction_first_player
        self.cards_first_player = cards_first_player

    def start_game(self, deck: Deck, players: List[Player], first_player):
        self._assign_deck(deck)
        self._assign_players(players)
        self._reveal_trump_card()
        self._distribute_cards()
        self.player_starting = first_player

    def _reveal_trump_card(self):
        indices_first_trump_card = 0
        while self.deck.cards[indices_first_trump_card].color != "RED":  # very dangerous
            indices_first_trump_card += 1
        self.trump_card = self.deck.cards[indices_first_trump_card]
        self.deck.cards = self.deck.cards[0:indices_first_trump_card] + self.deck.cards[indices_first_trump_card + 1:]

    def make_predictions(self):
        self.initial_predictions[self.players[0]] = self.prediction_first_player
        self.number_of_turns_won[self.players[0]] = 0
        for player in self.players[1:]:
            self.initial_predictions[player] = player.make_prediction()
            self.number_of_turns_won[player] = 0

    def _distribute_cards(self):
        assert self.players is not None, 'No players'
        assert self.deck is not None, 'Deck of cards is missing'
        assert self.number_of_cards_per_player * NUMBER_OF_PLAYERS < len(self.deck.cards), 'Not enough cards'

        self.players[0].cards = self.cards_first_player.copy()

        for player in self.players[1:]:
            player.cards = self.deck.cards[0:self.number_of_cards_per_player]
            self.deck.cards = self.deck.cards[self.number_of_cards_per_player:]
