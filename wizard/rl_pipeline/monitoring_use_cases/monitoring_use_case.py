import abc

from torch.utils.tensorboard import SummaryWriter


class MonitoringUseCase(abc.ABC):
    def __init__(self, frequency: int, writer: SummaryWriter, tensorboard_name: str | None = None):
        self.frequency = frequency
        self._writer = writer
        self._tensorboard_name = tensorboard_name

    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> None:
        pass
