from plotly import express as px

from config.common import NUMBER_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from wizard.base_game.player.card_play_policy import (
    DQNCardPlayPolicy,
    RandomCardPlayPolicy,
    StatisticalCardPlayPolicy,
)
from wizard.base_game.player.player import Player, RandomPlayer
from wizard.base_game.player.prediction_policy import (
    DefinedPredictionPolicy,
    DQNPredictionPolicy,
)
from wizard.exhaustive_simulation.simulation_result import SimulationResultMetadata
from wizard.exhaustive_simulation.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)
from wizard.exhaustive_simulation.survey_simulation_result import (
    transform_surveyed_df_to_have_predictions_as_index,
)
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.models.ann_pipeline import ANNPipeline, ANNSpecification

model = ANNPipeline(
    card_ann_specification=ANNSpecification(hidden_layers_size=[150, 150], output_size=10),
    hand_ann_specification=ANNSpecification(hidden_layers_size=[10, 10], output_size=10),
    strategy_ann_specification=ANNSpecification(hidden_layers_size=[10, 10], output_size=10),
    q_ann_specification=ANNSpecification(hidden_layers_size=[20, 20]),
)

agent = DQNAgent(model)

players = [
    Player(
        identifier=0,
        prediction_policy=DefinedPredictionPolicy,
        card_play_policy=DQNCardPlayPolicy,  # DQNCardPlayPolicy StatisticalCardPlayPolicy RandomCardPlayPolicy
        agent=agent,
        set_prediction=2,
        stat_table=transform_surveyed_df_to_have_predictions_as_index(
            SimulationResultStorage(
                simulation_result_metadata=SimulationResultMetadata(
                    simulation_id=195991601809031998,
                    learning_player_id=0,
                    number_of_players=NUMBER_OF_PLAYERS,
                    number_of_cards_per_player=NUMBER_CARDS_PER_PLAYER,
                    total_number_trial=1000,
                ),
                simulation_type=SimulationResultType.SURVEY,
            ).read_simulation_result()
        ),
    )
] + [RandomPlayer(i) for i in range(1, NUMBER_OF_PLAYERS)]
# players = [RandomPlayer(i) for i in range(NUMBER_OF_PLAYERS)]

env = SinglePlayerLearningEnv(players=players, starting_player=players[0], learning_player=players[0])

rewards = []
for _ in range(50_000):
    terminal = False
    state = env.reset()[0]
    while not terminal:
        next_state, reward, terminal, _, _ = env.step(None)
        if players[0].card_play_policy is DQNCardPlayPolicy or players[0].prediction_policy is DQNPredictionPolicy:
            agent.train(state, next_state, reward)
        state = next_state
    rewards.append(reward)

import numpy as np

freq = 5_000
mean_rewards = [np.mean(rewards[i * freq : (i + 1) * freq]) for i in range(len(rewards) // freq)]
px.line(mean_rewards).show()
