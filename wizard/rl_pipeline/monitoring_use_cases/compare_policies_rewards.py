from torch.utils.tensorboard import SummaryWriter

from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.deck import Deck
from wizard.base_game.player.player import Player, MaxRandomPlayer
from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase
from wizard.simulation.simulate_pre_defined_games import SimulatePreDefinedGames


class ComparePoliciesRewards(MonitoringUseCase):
    def __init__(
        self,
        frequency: int,
        writer: SummaryWriter,
        number_of_simulated_games: int,
        challenger_players_with_label: dict[str, Player],
        starting_player_position: int | None = None,
        tensorboard_name: str | None = None,
    ):
        super().__init__(frequency, writer, tensorboard_name)
        self._decks = [Deck() for _ in range(number_of_simulated_games)]
        self._challenger_players_with_label = challenger_players_with_label
        self._other_players = [MaxRandomPlayer(i) for i in range(1, NUMBER_OF_PLAYERS)]
        self._starting_player_position = starting_player_position

    def execute(self, epoch: int, *args, **kwargs) -> None:
        output = {}
        for name, challenger_player in self._challenger_players_with_label.items():
            players = [challenger_player] + self._other_players
            starting_player = players[self._starting_player_position] if self._starting_player_position else None
            output[name] = SimulatePreDefinedGames(
                decks=self._decks, players=players, learning_player=challenger_player, starting_player=starting_player
            ).execute()
        self._writer.add_scalars(self._tensorboard_name, output, epoch)
