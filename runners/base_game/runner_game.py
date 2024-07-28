from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game, GameDisplayer
from wizard.base_game.player.player import RandomPlayer

players = [RandomPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
deck = Deck()
game = Game(id_game=1)
game_displayer = GameDisplayer(game)

game.initialize_game(deck=deck, players=players, first_player=players[0])
game_displayer.display_hand_each_player()

game.request_predictions()
game_displayer.display_prediction_each_player()

game.play_game(print_results=True)

game_displayer.display_result_each_player()
