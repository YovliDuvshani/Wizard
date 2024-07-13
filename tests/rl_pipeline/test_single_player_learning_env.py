from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.player import RandomPlayer
from wizard.rl_pipeline.env.single_player_learning_env import \
    SinglePlayerLearningEnv


class TestSinglePlayerLearningEnv:
    def test_step_runs_through(self):
        players = [RandomPlayer(i) for i in range(NUMBER_OF_PLAYERS)]
        env = SinglePlayerLearningEnv(
            players=players, starting_player=players[0], learning_player=players[0]
        )
        results = []
        terminal = False
        results.append(env.reset()[0])
        while not terminal:
            action = players[0].playable_cards()[0].id
            state, reward, terminal, _, _ = env.step(action)
            results.append((state, reward, terminal))
        print()
