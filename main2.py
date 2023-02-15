import random

import numpy as np

from deck import Deck, FilteredDeck
from game import PredefinedGameForFirstPlayer
from hand_combinations import HandCombinationsTwoCards
from player import DecidePlayer

np.random.seed()
random.seed()

number_of_cards_to_play_with = 2

if __name__ == '__main__':
    deck = Deck(deck_id=1)
    hand_two_cards = HandCombinationsTwoCards(trump_color=deck.get_colors[0], base_color1=deck.get_colors[1],
                                              base_color2=deck.get_colors[2])
    possibilities_for_first_player_to_test = hand_two_cards.possibilities_to_test(deck)
    """
    record_possibilities_first_player = pd.DataFrame(possibilities_for_first_player_to_test,
                                                     columns=['first_card', 'second_card'])
    record_possibilities_first_player['first_card'] = record_possibilities_first_player['first_card'].apply(
        lambda x: [x.number, x.color] if x.number else x.special_card)
    record_possibilities_first_player['second_card'] = record_possibilities_first_player['second_card'].apply(
        lambda x: [x.number, x.color] if x.number else x.special_card)
    print(record_possibilities_first_player)
    """
    filtered_deck = FilteredDeck(deck_id=1, cards_to_suppress=possibilities_for_first_player_to_test[0])
    print(possibilities_for_first_player_to_test[0])

    player1 = DecidePlayer(id=1, first_card_to_play=0)
    player2 = DecidePlayer(id=2, first_card_to_play=0)
    player3 = DecidePlayer(id=3, first_card_to_play=0)

    players = [player1, player2, player3]
    game = PredefinedGameForFirstPlayer(id_game=1,
                                        number_of_cards_per_player=number_of_cards_to_play_with,
                                        prediction_first_player=2,
                                        cards_first_player=possibilities_for_first_player_to_test[0])

    game.start_game(deck, players, first_player=player1)
    game.make_predictions()

    game.play_game(log=True)

    for player in game.players:
        print(
            f'Player {player.id} annonced {game.initial_predictions[player]} won {game.number_of_turns_won[player]} got {game.count_points()[player]} points')
