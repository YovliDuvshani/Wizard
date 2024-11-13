from config.common import NUMBER_OF_PLAYERS
from wizard.base_game.player.player import RandomPlayer
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.models.multi_step_ann import MultiStepANN, ANNSpecification


players = [RandomPlayer(i) for i in range(NUMBER_OF_PLAYERS)]
env = SinglePlayerLearningEnv(players=players, starting_player=players[0], learning_player=players[0])
model = MultiStepANN(
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
    action = agent.select_action(state)
    next_state, reward, terminal, _, _ = env.step(action)
    results.append((next_state, reward, terminal))
    agent.train(state, action, next_state, reward)
    state = next_state
