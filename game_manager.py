import numpy as np
import pandas as pd

from deck import FilteredDeck, Deck
from game import PredefinedGameForFirstPlayer
from hand_combinations import HandCombinationsTwoCards
from player import DecidePlayer

NUMBER_CARDS_PER_PLAYER = 2


class GameManager:
    def __init__(self):
        pass


class GameManagerTwoCardsFirstPlayer(GameManager):
    def __init__(self, number_trial: int):  # strategy/ number_player
        self.number_trial = number_trial

    def simulate(self):
        summary = pd.DataFrame()
        deck = Deck(deck_id=1)
        hand_two_cards = HandCombinationsTwoCards(trump_color=deck.get_colors[0], base_color1=deck.get_colors[1],
                                                  base_color2=deck.get_colors[2])
        possibilities_for_first_player_to_test = hand_two_cards.possibilities_to_test(deck)
        for possibility in possibilities_for_first_player_to_test:
            filtered_deck = FilteredDeck(deck_id=1, cards_to_suppress=possibility)
            list_worst_points_gain_per_prediction_first_card_play = []
            list_worst_points_gain_per_prediction_second_card_play = []
            for trial in range(self.number_trial):
                worst_points_gain_per_prediction_first_card_play = {}
                worst_points_gain_per_prediction_second_card_play = {}
                filtered_deck.reset_deck_randomly()
                for prediction in range(NUMBER_CARDS_PER_PLAYER + 1):
                    game = PredefinedGameForFirstPlayer(id_game=1,
                                                        number_of_cards_per_player=2,
                                                        prediction_first_player=prediction,
                                                        cards_first_player=possibility.copy())
                    worst_points_gain_first_card_play = 1000
                    worst_points_gain_second_card_play = 1000
                    for player_1_first_card_to_play in [0, 1]:
                        for player_2_first_card_to_play in [0, 1]:
                            for player_3_first_card_to_play in [0, 1]:
                                player1 = DecidePlayer(id=1, first_card_to_play=player_1_first_card_to_play)
                                player2 = DecidePlayer(id=2, first_card_to_play=player_2_first_card_to_play)
                                player3 = DecidePlayer(id=3, first_card_to_play=player_3_first_card_to_play)
                                players = [player1, player2, player3]
                                filtered_deck.reset_deck()
                                game.start_game(filtered_deck, players, first_player=player1)
                                game.make_predictions()
                                game.play_game(log=False)

                                points = game.count_points()[game.players[0]]
                                if (player_1_first_card_to_play == 0) & (points < worst_points_gain_first_card_play):
                                    worst_points_gain_first_card_play = points
                                elif (player_1_first_card_to_play == 1) & (points < worst_points_gain_second_card_play):
                                    worst_points_gain_second_card_play = points
                    worst_points_gain_per_prediction_first_card_play[prediction] = worst_points_gain_first_card_play
                    worst_points_gain_per_prediction_second_card_play[prediction] = worst_points_gain_second_card_play
                list_worst_points_gain_per_prediction_first_card_play += [
                    worst_points_gain_per_prediction_first_card_play]
                list_worst_points_gain_per_prediction_second_card_play += [
                    worst_points_gain_per_prediction_second_card_play]
            mean_per_prediction_first_card_play = {}
            mean_per_prediction_second_card_play = {}
            for prediction in range(3):
                aux = []
                for ele in list_worst_points_gain_per_prediction_first_card_play:
                    aux += [ele[prediction]]
                mean_per_prediction_first_card_play[prediction] = np.mean(aux)
            for prediction in range(3):
                aux = []
                for ele in list_worst_points_gain_per_prediction_second_card_play:
                    aux += [ele[prediction]]
                mean_per_prediction_second_card_play[prediction] = np.mean(aux)
            summary = summary.append({'first_card': possibility[0].prepare_card_for_display(),
                                      'second_card': possibility[1].prepare_card_for_display(),
                                      'prediction_0_worst_average_gain_first_card_play':
                                          mean_per_prediction_first_card_play[0],
                                      'prediction_1_worst_average_gain_first_card_play':
                                          mean_per_prediction_first_card_play[1],
                                      'prediction_2_worst_average_gain_first_card_play':
                                          mean_per_prediction_first_card_play[2],
                                      'prediction_0_worst_average_gain_second_card_play':
                                          mean_per_prediction_second_card_play[0],
                                      'prediction_1_worst_average_gain_second_card_play':
                                          mean_per_prediction_second_card_play[1],
                                      'prediction_2_worst_average_gain_second_card_play':
                                          mean_per_prediction_second_card_play[2],
                                      }, ignore_index=True)
        summary['best_outcome_prediction_0'] = summary[['prediction_0_worst_average_gain_first_card_play',
                                                        'prediction_0_worst_average_gain_second_card_play']].values.max(
            1)
        summary['best_outcome_prediction_1'] = summary[['prediction_1_worst_average_gain_first_card_play',
                                                        'prediction_1_worst_average_gain_second_card_play']].values.max(
            1)
        summary['best_outcome_prediction_2'] = summary[['prediction_2_worst_average_gain_first_card_play',
                                                        'prediction_2_worst_average_gain_second_card_play']].values.max(
            1)
        summary = summary[['first_card', 'second_card', 'best_outcome_prediction_0', 'best_outcome_prediction_1',
                           'best_outcome_prediction_2']]
        summary.to_csv('2_cards_everybody_against_you')
