from config.common import NUMBER_OF_PLAYERS, NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.player.player import RandomPlayer
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.models.ann_pipeline import ANNPipeline, ANNSpecification


class TestSinglePlayerLearningEnv:
    def test_step_runs_through(self):
        players = [RandomPlayer(i) for i in range(NUMBER_OF_PLAYERS)]
        env = SinglePlayerLearningEnv(players=players, starting_player=players[0], learning_player=players[0])
        model = ANNPipeline(
            card_ann_specification=ANNSpecification(hidden_layers_size=[150, 150], output_size=10),
            hand_ann_specification=ANNSpecification(hidden_layers_size=[10, 10], output_size=10),
            strategy_ann_specification=ANNSpecification(hidden_layers_size=[10, 10], output_size=10),
            q_ann_specification=ANNSpecification(hidden_layers_size=[20, 20]),
        )
        agent = DQNAgent(model)
        results = []
        terminal = False
        state = env.reset()[0]
        results.append(state)
        while not terminal:
            next_state, reward, terminal, _, _ = env.step(None)
            results.append((next_state, reward, terminal))
            agent.train(state, next_state, reward)
            state = next_state
        print()
