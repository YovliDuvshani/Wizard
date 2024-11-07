from copy import deepcopy

from config.common import NUMBER_OF_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.player.player import DQNPlayer, RandomPlayer
from wizard.rl_pipeline.features.compute_generic_features import ComputeGenericFeatures
from wizard.rl_pipeline.features.select_learning_features import SelectLearningFeatures
from wizard.simulation.exhaustive.hand_combinations import IMPLEMENTED_COMBINATIONS
from wizard.simulation.exhaustive.simulator import CombinationNotImplemented


class ComputeQValueAllHandCombinationsFirstPlayer:
    def __init__(self, player: DQNPlayer, deck: Deck = Deck()):
        if NUMBER_OF_CARDS_PER_PLAYER not in IMPLEMENTED_COMBINATIONS:
            raise CombinationNotImplemented
        self._player = player
        self._hand_combinations_class = IMPLEMENTED_COMBINATIONS[NUMBER_OF_CARDS_PER_PLAYER]
        self._initial_deck = deck

    def execute(self):
        q_values = {}
        hand_combinations = self._hand_combinations_class(
            deck=self._initial_deck
        ).build_all_possible_hand_combinations()
        for combination in hand_combinations:
            deck = deepcopy(self._initial_deck)
            deck.remove_cards(cards_to_remove=combination)
            self._player.receive_cards(combination)
            game = Game()
            game.initialize_game(
                deck=deck,
                players=[RandomPlayer(0)] + [self._player] + [RandomPlayer(i) for i in range(2, NUMBER_OF_PLAYERS)],
                starting_player=self._player,
            )
            combination_representation = combination[0].representation  # TODO: Generalize to 2+ cards
            q_values[combination_representation] = self._player.agent.q_max(
                SelectLearningFeatures().execute(ComputeGenericFeatures(self._player.game, self._player).execute())
            )
            self._player.drop_hand()
        return q_values
