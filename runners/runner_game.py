from config.common import NUMBER_OF_PLAYERS
from wizard.deck import Deck
from wizard.game import Game
from wizard.player import RandomPlayer

players = [RandomPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
deck = Deck()
game = Game(id_game=1)

game.initialize_game(deck=deck, players=players, first_player=players[0])
game.display_hand_each_player()

game.request_predictions()
game.display_prediction_each_player()

game.play_round(print_results=True)

game.display_result_each_player()
