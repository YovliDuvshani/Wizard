import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import torch
from plotly import express as px
from pyinstrument import Profiler

from config.common import NUMBER_CARDS_PER_PLAYER, NUMBER_OF_PLAYERS
from project_path import ABS_PATH_PROJECT
from wizard.base_game.player.card_play_policy import (
    DQNCardPlayPolicy,
    HighestCardPlayPolicy,
    LowestCardPlayPolicy,
    RandomCardPlayPolicy,
    StatisticalCardPlayPolicy,
)
from wizard.base_game.player.player import MaxRandomPlayer, Player, RandomPlayer
from wizard.base_game.player.prediction_policy import (
    DefinedPredictionPolicy,
    DQNPredictionPolicy,
)
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.models.ann_pipeline import ANNPipeline, ANNSpecification
from wizard.simulation.exhaustive.simulation_result import SimulationResultMetadata
from wizard.simulation.exhaustive.simulation_result_storage import (
    SimulationResultStorage,
    SimulationResultType,
)
from wizard.simulation.exhaustive.survey_simulation_result import (
    transform_surveyed_df_to_have_predictions_as_index,
)
from wizard.simulation.simulate_pre_defined_games import SimulatePreDefinedGames

model = ANNPipeline(
    card_ann_specification=ANNSpecification(hidden_layers_size=[500, 500, 500], output_size=100),
    hand_ann_specification=ANNSpecification(hidden_layers_size=[100, 100], output_size=100),
    strategy_ann_specification=ANNSpecification(hidden_layers_size=[500, 500, 500], output_size=100),
    q_ann_specification=ANNSpecification(hidden_layers_size=[100, 100, 100]),
)

agent = DQNAgent(model)

players = [
    Player(
        identifier=0,
        prediction_policy=DefinedPredictionPolicy,
        card_play_policy=DQNCardPlayPolicy,
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
players_per_defined_simulation = [players[0]] + [MaxRandomPlayer(i) for i in range(1, NUMBER_OF_PLAYERS)]

env = SinglePlayerLearningEnv(players=players, starting_player=players[0], learning_player=players[0])

simulate_pre_defined_games_test_set = SimulatePreDefinedGames(num_games=500)
simulate_pre_defined_games_test_set.assign_players(
    players=players_per_defined_simulation, target_player=players[0], player_starting=players[0]
)

simulate_pre_defined_games_val_set = SimulatePreDefinedGames(num_games=500)
simulate_pre_defined_games_val_set.assign_players(
    players=players_per_defined_simulation, target_player=players[0], player_starting=players[0]
)

profiler = Profiler()
profiler.start()

run_id = uuid.uuid4()

rewards_validation_games = {}
rewards_test_games = {}
for i in range(150_001):
    if i % 2000 == 0:
        agent.set_deterministic_action_choice(True)
        rewards_validation_games[i] = simulate_pre_defined_games_val_set.execute()
        rewards_test_games[i] = simulate_pre_defined_games_test_set.execute()
        agent.set_deterministic_action_choice(False)
        print(f"Iteration {i}")
        model_path = f"{ABS_PATH_PROJECT}/trained_models/run={run_id}/iteration={i}"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        if i % 20000 == 0:
            best_model_path = f"{ABS_PATH_PROJECT}/trained_models/run={run_id}/iteration={max(rewards_validation_games, key=rewards_validation_games.get)}"
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            agent._loss = torch.tensor(0, dtype=float)

    terminal = False
    state = env.reset()[0]
    while not terminal:
        next_state, reward, terminal, _, _ = env.step(None)
        if players[0].card_play_policy is DQNCardPlayPolicy or players[0].prediction_policy is DQNPredictionPolicy:
            agent.train(state, next_state, reward)
        state = next_state

challenger_players = {
    name: Player(
        identifier=0,
        prediction_policy=DefinedPredictionPolicy,
        card_play_policy=card_play_policy,
        name=name,
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
    for name, card_play_policy in [
        ("optimal", StatisticalCardPlayPolicy),
        ("highest_card", HighestCardPlayPolicy),
        ("lowest_card", LowestCardPlayPolicy),
        ("random", RandomCardPlayPolicy),
    ]
}

profiler.stop()
profiler.open_in_browser()

challenger_results = {}
for name, player in challenger_players.items():
    simulate_pre_defined_games_test_set.assign_players(
        players=[player] + players_per_defined_simulation[1:], target_player=player, player_starting=player
    )
    challenger_results[name] = simulate_pre_defined_games_test_set.execute()

fig = px.line(pd.DataFrame.from_dict(rewards_test_games, orient="index", columns=["reward_dqn"]), title="test")
for name, result in challenger_results.items():
    fig.add_trace(
        go.Scatter(y=[result for i in range(max(rewards_test_games.keys()))], mode="lines", name=f"reward_{name}")
    )
fig.show()

challenger_results = {}
for name, player in challenger_players.items():
    simulate_pre_defined_games_val_set.assign_players(
        players=[player] + players_per_defined_simulation[1:], target_player=player, player_starting=player
    )
    challenger_results[name] = simulate_pre_defined_games_val_set.execute()

fig = px.line(
    pd.DataFrame.from_dict(rewards_validation_games, orient="index", columns=["reward_dqn"]), title="validation"
)
for name, result in challenger_results.items():
    fig.add_trace(
        go.Scatter(y=[result for i in range(max(rewards_validation_games.keys()))], mode="lines", name=f"reward_{name}")
    )
fig.show()
