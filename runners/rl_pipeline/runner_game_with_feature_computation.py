from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.count_points import CountPoints
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game, GameDisplayer
from wizard.base_game.player import RandomPlayer
from wizard.rl_pipeline.features.compute_generic_features import ComputeGenericFeatures

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

for _ in range(1000):
    players = [RandomPlayer(identifier=i) for i in range(NUMBER_OF_PLAYERS)]
    learning_player = players[1]
    deck = Deck()
    game = Game(id_game=1)
    game_displayer = GameDisplayer(game)

    game.initialize_game(deck=deck, players=players, first_player=players[0])
    game_displayer.display_hand_each_player()

    game.request_predictions()
    game_displayer.display_prediction_each_player()

    game.get_to_first_state_for_given_player(learning_player)

    terminal = game.get_to_next_afterstate_for_given_player(
        learning_player, print_results=True
    )
    while not terminal:
        features = ComputeGenericFeatures(game, learning_player).execute()
        terminal = game.get_to_next_afterstate_for_given_player(
            learning_player, print_results=True
        )
        reward = CountPoints().execute(
            game.state.predictions, game.state.number_of_turns_won
        )

profiler.stop()
profiler.open_in_browser()
