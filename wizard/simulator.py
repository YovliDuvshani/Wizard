import abc
import itertools
from copy import deepcopy
from typing import List, Optional
import datetime as dt

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.card import Card
from wizard.common import iterator_to_list_of_list
from wizard.deck import Deck
from wizard.game import Game
from wizard.hand_combinations import HandCombinationsTwoCards
from wizard.player import Player
from wizard.simulation_result import SimulationResult

IMPLEMENTED_COMBINATIONS = {2: HandCombinationsTwoCards}


class Simulator(abc.ABC):
    def __init__(
        self,
        players: List[Player],
        initial_deck: Deck,
        simulation_id: Optional[int] = dt.datetime.today().__hash__(),
    ):
        self.initial_deck = initial_deck
        self.players = (
            players  # the order of players defines the starting playing order
        )
        self.simulation_id = simulation_id

    def simulate(self):
        pass


class SimulatorWithOneLearningPlayer(Simulator):
    def __init__(
        self,
        players: List[Player],
        learning_player: Player,
        initial_deck: Deck,
        number_trial_each_combination: int,
    ):
        super().__init__(players=players, initial_deck=initial_deck)
        if learning_player not in players:
            raise LearningPlayerNotPlaying
        self.learning_player = learning_player
        self.number_trial_each_combination = number_trial_each_combination
        if NUMBER_CARDS_PER_PLAYER not in IMPLEMENTED_COMBINATIONS.keys():
            raise CombinationNotImplemented
        self.hand_combinations_class = IMPLEMENTED_COMBINATIONS[NUMBER_CARDS_PER_PLAYER]
        self.result_logger: Optional[List[SimulationResult]] = None
        self.first_player = players[0]

    def simulate(self):
        self.result_logger = []
        hand_combinations = self.hand_combinations_class(
            deck=self.initial_deck
        ).build_hand_combinations()
        for combination in hand_combinations:
            deck = deepcopy(self.initial_deck)
            deck.remove_cards(cards_to_suppress=combination)
            self.assign_cards_to_learning_player(combination)
            for trial_number in range(self.number_trial_each_combination):
                self.reset_only_learning_player()
                deck.shuffle()
                game = Game()
                game.initialize_game(
                    deck=deepcopy(deck),
                    players=self.players,
                    first_player=self.first_player,
                )
                self.simulate_all_outcome_one_round(
                    game=game, trial_number=trial_number, combination=combination
                )
        return self.result_logger

    def simulate_all_outcome_one_round(
        self, game: Game, trial_number: int, combination: List[Card]
    ):
        all_playing_order_per_player: List[List[List[Card]]] = [
            iterator_to_list_of_list(itertools.permutations(player.cards))
            for player in self.players
        ]
        all_playing_order: List[List[List[Card]]] = iterator_to_list_of_list(
            itertools.product(*all_playing_order_per_player)
        )
        for playing_order in all_playing_order:
            self.simulate_one_outcome_one_round(
                game=game,
                playing_order=playing_order,
                trial_number=trial_number,
                combination=combination,
            )

    def simulate_one_outcome_one_round(
        self,
        game: Game,
        playing_order: List[List[Card]],
        trial_number: int,
        combination: List[Card],
    ):
        game.reset_game()
        for player, playing_order_per_player in zip(self.players, playing_order):
            if (
                player == self.learning_player
            ):  # Workaround could be removed if combinations are handled more elegantly
                combination_learning_player_order = playing_order_per_player
            player.provide_strategy(cards_ordered_by_priority=playing_order_per_player)
        game.play_round()
        self.result_logger.append(
            SimulationResult(
                trial_number=trial_number,
                tested_combination=" - ".join(
                    list(map(lambda card: card.representation, combination))
                ),
                combination_played_order=" - ".join(
                    list(
                        map(
                            lambda card: card.representation,
                            combination_learning_player_order,
                        )
                    )
                ),
                number_of_turns_won={
                    player.identifier: game.number_of_turns_won[player]
                    for player in self.players
                },
            )
        )

    def assign_cards_to_learning_player(self, combination: List[Card]) -> None:
        self.learning_player.receive_cards(combination)

    def reset_only_learning_player(self) -> None:
        self.learning_player.reset_hand()


class CombinationNotImplemented(Exception):
    pass


class LearningPlayerNotPlaying(Exception):
    pass
