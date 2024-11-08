import uuid
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from project_path import ABS_PATH_PROJECT
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase


class SaveModel(MonitoringUseCase):
    def __init__(
        self,
        frequency: int,
        writer: SummaryWriter,
        agent: DQNAgent,
        run_id: uuid.UUID,
        tensorboard_name: str | None = None,
    ):
        super().__init__(frequency, writer, tensorboard_name)
        self._agent = agent
        self._run_id = run_id

    def execute(self, epoch: int, *args, **kwargs) -> None:
        path = self._get_path(epoch)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._agent.model.state_dict(), path
        )  # TO LOAD BACK: model.load_state_dict(torch.load(path, weights_only=True))

    def _get_path(self, epoch: int):
        return f"{ABS_PATH_PROJECT}/trained_models/run={self._run_id}/iteration={epoch}"
