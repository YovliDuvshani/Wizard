from torch.utils.tensorboard import SummaryWriter

from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase


class LogLoss(MonitoringUseCase):
    def __init__(self, frequency: int, writer: SummaryWriter, tensorboard_name: str | None = "Train Loss"):
        super().__init__(frequency, writer, tensorboard_name)

    def execute(self, epoch: int, loss: float, *args, **kwargs) -> None:
        self._writer.add_scalar(self._tensorboard_name, loss, epoch)
