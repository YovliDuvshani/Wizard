import os
import shutil
import uuid

from torch.utils.tensorboard import SummaryWriter

from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.player.card_play_policy import (
    DQNCardPlayPolicy,
    StatisticalCardPlayPolicy,
    HighestCardPlayPolicy,
    LowestCardPlayPolicy,
    RandomCardPlayPolicy,
)
from wizard.base_game.player.player import Player, RandomPlayer
from wizard.base_game.player.prediction_policy import (
    DefinedPredictionPolicy,
)
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.constants import TENSORBOARD_OUTPUT_DIRECTORY
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.models.multi_step_ann import ANNSpecification, MultiStepANN
from wizard.rl_pipeline.monitoring_use_cases.compare_policies_rewards import ComparePoliciesRewards
from wizard.rl_pipeline.monitoring_use_cases.compute_q_values_statistics import ComputeQValuesStatistics
from wizard.rl_pipeline.monitoring_use_cases.create_ann_graph import CreateANNGraph
from wizard.rl_pipeline.monitoring_use_cases.log_loss import LogLoss
from wizard.rl_pipeline.monitoring_use_cases.save_model import SaveModel
from wizard.rl_pipeline.train_pipeline.base_train_pipeline import BaseTrainPipeline

model = MultiStepANN(
    card_ann_specification=ANNSpecification(hidden_layers_size=[500, 500, 500], output_size=100),
    hand_ann_specification=ANNSpecification(hidden_layers_size=[50, 50], output_size=20),
    strategy_ann_specification=ANNSpecification(hidden_layers_size=[200, 200], output_size=100),
    q_ann_specification=ANNSpecification(hidden_layers_size=[500, 500, 500]),
)
# model = BaseANN(ANNSpecification(hidden_layers_size=[1000, 1000]))

agent = DQNAgent(model)

learning_player = Player(
    identifier=0,
    prediction_policy=DefinedPredictionPolicy,  # DQNPredictionPolicy
    card_play_policy=DQNCardPlayPolicy,
    agent=agent,
    set_prediction=1,
)
players = [learning_player] + [RandomPlayer(i) for i in range(1, NUMBER_OF_PLAYERS)]
starting_player = None

env = SinglePlayerLearningEnv(players=players, learning_player=learning_player, starting_player=starting_player)

NUMBER_OF_EPOCHS = 500_000

# profiler = Profiler()
# profiler.start()

if os.path.exists(TENSORBOARD_OUTPUT_DIRECTORY):
    shutil.rmtree(TENSORBOARD_OUTPUT_DIRECTORY)
writer = SummaryWriter(TENSORBOARD_OUTPUT_DIRECTORY)

challenger_players_with_label = {
    name: Player(
        identifier=0,
        prediction_policy=DefinedPredictionPolicy,
        card_play_policy=card_play_policy,
        name=name,
        agent=agent,
        set_prediction=1,
    )
    for name, card_play_policy in [
        ("optimal", StatisticalCardPlayPolicy),
        ("highest_card", HighestCardPlayPolicy),
        ("lowest_card", LowestCardPlayPolicy),
        ("random", RandomCardPlayPolicy),
        ("dqn", DQNCardPlayPolicy),
    ]
}
starting_player_position = players.index(starting_player) if starting_player else None
monitoring_use_cases = [
    CreateANNGraph(frequency=NUMBER_OF_EPOCHS + 1, writer=writer, env=env, agent=agent),
    ComputeQValuesStatistics(frequency=10_000, writer=writer, env=env, agent=agent, number_of_games=500),
    LogLoss(frequency=1, writer=writer),
    SaveModel(frequency=10_000, writer=writer, agent=agent, run_id=uuid.uuid4()),
    ComparePoliciesRewards(
        frequency=2_000,
        writer=writer,
        number_of_simulated_games=100,
        challenger_players_with_label=challenger_players_with_label,
        starting_player_position=starting_player_position,
        tensorboard_name="Avg Reward on Validation Set",
    ),
    ComparePoliciesRewards(
        frequency=2_000,
        writer=writer,
        number_of_simulated_games=100,
        challenger_players_with_label=challenger_players_with_label,
        starting_player_position=starting_player_position,
        tensorboard_name="Avg Reward on Test Set",
    ),
]

BaseTrainPipeline(agent=agent, env=env, monitoring_use_cases=monitoring_use_cases).execute(
    number_of_epoch=NUMBER_OF_EPOCHS
)

writer.close()

# profiler.stop()
# profiler.open_in_browser()
