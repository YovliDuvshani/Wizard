import os
import random
import time

import numpy as np

from deck import Deck
from game import Game
from player import RandomPlayer

np.random.seed()
random.seed()

number_of_cards_to_play_with = 5

if __name__ == '__main__':
    deck = Deck(deck_id=1)
    player1 = RandomPlayer(1)
    player2 = RandomPlayer(2)
    player3 = RandomPlayer(3)

    players = [player1, player2, player3]
    game = Game(id_game=1,
                number_of_cards_per_player=number_of_cards_to_play_with)

    game.start_game(deck, players, first_player=player1)
    game.make_predictions()

    print(f'The Trump color is {game.trump_card.color}')

    # Cards of each player
    for player in game.players:
        print(f'Player {player.id} initial hand:',end='')
        for card in player.cards:
            print(f' {card.prepare_card_for_display()} ',end='--')
        print()

    game.play_game(log=True)

    for player in game.players:
        print(f'Player {player.id} annonced {game.initial_predictions[player]} won {game.number_of_turns_won[player]} got {game.count_points()[player]} points')

    start_time = time.time()
    for i in range(1):
        deck = Deck(deck_id=1)
        game.start_game(deck, players, first_player=player1)
        game.make_predictions()
        game.play_game(log=False)
    print(time.time() - start_time)

    two_cards_num_possibilities = (17*15/2)+(15*13)+(13*12/2)+13*13
    three_cards_num_possibilities = (17*16*15/6+17*16*13/2+17*(13*12/2+13*13)+13*12*11/6+13*13*12/2+13**3)

    print(two_cards_num_possibilities)
    print(three_cards_num_possibilities)
    print(two_cards_num_possibilities*1000*8*4)
    print(three_cards_num_possibilities*100*27*4)

    print(os.getcwd())
    print(None == None)