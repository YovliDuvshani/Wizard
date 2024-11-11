from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import io
from PIL import Image

from wizard.base_game.card import Card
from wizard.rl_pipeline.agents.DQNAgent import DQNAgent
from wizard.rl_pipeline.constants import BASE_WIDTH, BASE_HEIGHT
from wizard.rl_pipeline.env.single_player_learning_env import SinglePlayerLearningEnv
from wizard.rl_pipeline.features.data_cls import GenericFeatures
from wizard.rl_pipeline.monitoring_use_cases.monitoring_use_case import MonitoringUseCase

import plotly.express as px
import plotly.graph_objects as go


@dataclass
class QValuesPlotDefinition:
    plot_func: Callable[..., go.Figure]
    x: str | None = None
    y: str | None = None
    color: str | None = None
    facet_row: str | None = None
    facet_col: str | None = None
    sorting_cols: list[str] | None = None
    filtering_with_name: tuple[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None

    @property
    def name(self):
        return (
            f"{self.plot_func.__name__}_f({self.x})={self.y}_with_color_{self.color}_per_{self.facet_row}"
            f"_with_filtering_{self.filtering_with_name[0]}"
        )


Q_VALUES_PLOT_DEFINITIONS = [
    QValuesPlotDefinition(
        plot_func=px.scatter,
        x="FIRST_CARD_REPRESENTATION",
        y="Q_VALUES",
        facet_col="NUMBER_ROUNDS_TO_WIN",
        facet_row="PLAYER_POSITION",
        sorting_cols=["FIRST_CARD"],
        filtering_with_name=("only_prediction_step", lambda df: df[df["IS_PREDICTION_STEP"] == 1]),
    ),
    QValuesPlotDefinition(
        plot_func=px.scatter,
        x="FIRST_CARD_REPRESENTATION",
        y="Q_VALUES",
        facet_row="PLAYER_POSITION",
        facet_col="CAN_WIN_CURRENT_SUB_ROUND",
        sorting_cols=["FIRST_CARD"],
        filtering_with_name=(
            "prediction=0",
            lambda df: df[(df["IS_PREDICTION_STEP"] == -1) & (df["NUMBER_ROUNDS_TO_WIN"] == 0)],
        ),
    ),
    QValuesPlotDefinition(
        plot_func=px.scatter,
        x="FIRST_CARD_REPRESENTATION",
        y="Q_VALUES",
        facet_row="PLAYER_POSITION",
        facet_col="CAN_WIN_CURRENT_SUB_ROUND",
        sorting_cols=["FIRST_CARD"],
        filtering_with_name=(
            "prediction=1",
            lambda df: df[(df["IS_PREDICTION_STEP"] == -1) & (df["NUMBER_ROUNDS_TO_WIN"] == 1)],
        ),
    ),
]


class ComputeQValuesStatistics(MonitoringUseCase):
    def __init__(
        self,
        frequency: int,
        writer: SummaryWriter,
        env: SinglePlayerLearningEnv,
        agent: DQNAgent,
        number_of_games: int,
        tensorboard_name: str | None = None,
    ):
        super().__init__(frequency, writer, tensorboard_name)
        self._env = env
        self._agent = agent
        self._number_of_games = number_of_games

    def execute(self, epoch: int, *args, **kwargs) -> None:
        states, q_values = [], []
        for _ in range(self._number_of_games):
            terminal = False
            state = self._env.reset()[0]
            while not terminal:
                action = self._agent.select_action(state)
                state = self._agent.update_state_to_action_state_for_prediction_phase(state, action)

                states.append(state)
                q_values.append(self._agent.q_max(state))

                next_state, reward, terminal, _, _ = self._env.step(action)
                state = next_state

        feature_df_with_q = self._build_feature_df_with_q(states, q_values)
        self._create_and_log_plots(feature_df_with_q, epoch)

    @staticmethod
    def _build_feature_df_with_q(states: list[GenericFeatures], q_values: list[float]):
        feature_df_with_q = pd.DataFrame(
            [{**state.generic_cards_context.__dict__, **state.generic_objective_context.__dict__} for state in states]
        )
        feature_df_with_q["Q_VALUES"] = q_values
        feature_df_with_q["CAN_WIN_CURRENT_SUB_ROUND"] = [
            list(state.generic_card_specific.values())[0].CAN_WIN_CURRENT_SUB_ROUND for state in states
        ]
        feature_df_with_q["FIRST_CARD_REPRESENTATION"] = [
            list(state.generic_card_specific.keys())[0] for state in states
        ]
        feature_df_with_q["FIRST_CARD"] = feature_df_with_q["FIRST_CARD_REPRESENTATION"].apply(
            lambda representation: Card.from_representation(representation)
        )
        return feature_df_with_q

    def _create_and_log_plots(self, feature_df_with_q: pd.DataFrame, epoch: int):
        for plot_def in Q_VALUES_PLOT_DEFINITIONS:
            fig = self._create_plot(plot_def, feature_df_with_q)
            self._log_plot_in_tensorboard(fig, epoch, plot_def.name)

    @staticmethod
    def _create_plot(plot_def: QValuesPlotDefinition, feature_df_with_q: pd.DataFrame) -> go.Figure:
        kwargs = {
            "x": plot_def.x,
            "y": plot_def.y,
            "color": plot_def.color,
            "facet_row": plot_def.facet_row,
            "facet_col": plot_def.facet_col,
            "width": BASE_WIDTH,
            "height": BASE_HEIGHT,
        }
        if plot_def.filtering_with_name is not None:
            feature_df_with_q = feature_df_with_q.copy()
            feature_df_with_q = plot_def.filtering_with_name[1](feature_df_with_q)
        if plot_def.sorting_cols:
            feature_df_with_q = feature_df_with_q.sort_values(plot_def.sorting_cols)
        return plot_def.plot_func(feature_df_with_q, **kwargs)

    def _log_plot_in_tensorboard(self, fig: go.Figure, epoch: int, name: str):
        image_bytes = fig.to_image(format="png")
        buf = io.BytesIO(image_bytes)
        img = Image.open(buf)
        self._writer.add_image(name, np.asarray(img), epoch, dataformats="HWC")
