# mypy: disable-error-code="union-attr"
import abc
import datetime as dt
import itertools
from copy import deepcopy
from typing import List, Optional

from config.common import NUMBER_CARDS_PER_PLAYER
from wizard.base_game.card import Card
from wizard.base_game.deck import Deck
from wizard.base_game.game import Game
from wizard.base_game.list_cards import ListCards
from wizard.base_game.player import DefinedStrategyPlayer
from wizard.common import iterator_to_list_of_list
from wizard.exhaustive_simulation.hand_combinations import HandCombinationsTwoCards
from wizard.exhaustive_simulation.simulation_result import SimulationResult

IMPLEMENTED_COMBINATIONS = {2: HandCombinationsTwoCards}


class Simulator(abc.ABC):
    def __init__(
        self,
        players: List[DefinedStrategyPlayer],
        initial_deck: Deck,
        simulation_id: Optional[int] = dt.datetime.today().__hash__(),
    ):
        self.simulation_id = simulation_id
        self._initial_deck = initial_deck
        self._players = players

    def simulate(self):
        pass


class SimulatorWithOneLearningPlayer(Simulator):
    def __init__(
        self,
        players: List[DefinedStrategyPlayer],
        learning_player: DefinedStrategyPlayer,
        initial_deck: Deck,
        number_trial_each_combination: int,
    ):
        if NUMBER_CARDS_PER_PLAYER not in IMPLEMENTED_COMBINATIONS:
            raise CombinationNotImplemented
        if learning_player not in players:
            raise LearningPlayerNotPlaying
        super().__init__(players=players, initial_deck=initial_deck)
        self._learning_player = learning_player
        self._number_trial_each_combination = number_trial_each_combination
        self._hand_combinations_class = IMPLEMENTED_COMBINATIONS[
            NUMBER_CARDS_PER_PLAYER
        ]

    def simulate(self):
        result_logger: List[SimulationResult] = []
        hand_combinations = self._hand_combinations_class(
            deck=self._initial_deck
        ).build_all_possible_hand_combinations()
        for combination in hand_combinations:
            deck = deepcopy(self._initial_deck)
            deck.remove_cards(cards_to_suppress=combination)
            self._learning_player.receive_cards(combination)
            for trial_number in range(self._number_trial_each_combination):
                self._learning_player.reset_hand()
                deck.shuffle()
                game = Game()
                game.initialize_game(
                    deck=deck,
                    players=self._players,
                    first_player=self._players[0],
                )
                self._simulate_all_outcome_one_round(
                    game=game,
                    trial_number=trial_number,
                    result_logger=result_logger,
                )
                game.definition.deck.reset_deck()
        return result_logger

    def _simulate_all_outcome_one_round(
        self, game: Game, trial_number: int, result_logger: List[SimulationResult]
    ):
        all_playing_order_per_player: List[List[List[Card]]] = [
            iterator_to_list_of_list(itertools.permutations(player.cards))
            for player in self._players
        ]
        all_playing_order: List[List[List[Card]]] = iterator_to_list_of_list(
            itertools.product(*all_playing_order_per_player)
        )
        for playing_order in all_playing_order:
            self._simulate_one_outcome_one_round(
                game=game,
                playing_order=playing_order,
                trial_number=trial_number,
                result_logger=result_logger,
            )

    def _simulate_one_outcome_one_round(
        self,
        game: Game,
        playing_order: List[List[Card]],
        trial_number: int,
        result_logger: List[SimulationResult],
    ):
        for i, player in enumerate(self._players):
            player.provide_strategy(cards_ordered_by_priority=playing_order[i])
        game.reset_game()
        game.play_game()
        result_logger.append(
            SimulationResult(
                trial_number=trial_number,
                tested_combination=ListCards(
                    cards=playing_order[self._players.index(self._learning_player)]
                ).to_single_representation(sort=True),
                combination_played_order=ListCards(
                    cards=playing_order[self._players.index(self._learning_player)]
                ).to_single_representation(sort=False),
                number_of_turns_won={
                    player.identifier: game.state.number_of_turns_won[player]
                    for player in self._players
                },
            )
        )


class CombinationNotImplemented(Exception):
    pass


class LearningPlayerNotPlaying(Exception):
    pass
