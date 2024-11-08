from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase


class BaseTrainPipeline:
    def __init__(
        self,
        agent: DQNAgent,
        env: SinglePlayerLearningEnv,
        monitoring_use_cases: list[MonitoringUseCase],
    ):
        self._agent = agent
        self._env = env
        self._monitoring_use_cases = monitoring_use_cases

    def execute(self, number_of_epoch: int):
        for epoch in range(number_of_epoch + 1):
            if epoch % 1_000 == 0:
                print(f"Iteration {epoch}")
            terminal = False
            state = self._env.reset()[0]
            while not terminal:
                next_state, reward, terminal, _, _ = self._env.step(None)
                loss = self._agent.train(state, next_state, reward)
                state = next_state
            self._run_monitoring_use_cases(epoch, loss, reward)

    def _run_monitoring_use_cases(self, epoch: int, loss: float, reward: float):
        for monitoring_use_case in self._monitoring_use_cases:
            if epoch % monitoring_use_case.frequency == 0:
                monitoring_use_case.execute(epoch=epoch, loss=loss, reward=reward)
