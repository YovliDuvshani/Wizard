from torch.utils.tensorboard import SummaryWriter

from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase


class CreateANNGraph(MonitoringUseCase):
    def __init__(
        self,
        frequency: int,
        writer: SummaryWriter,
        env: SinglePlayerLearningEnv,
        agent: DQNAgent,
        tensorboard_name: str | None = None,
    ):
        super().__init__(frequency, writer, tensorboard_name)
        self._env = env
        self._agent = agent

    def execute(self, *args, **kwargs) -> None:
        state = self._env.reset()[0]
        state_tensor = self._agent.convert_array_to_tensor(state)
        self._writer.add_graph(self._agent.model, state_tensor)
