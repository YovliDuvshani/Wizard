import random
from copy import deepcopy

import numpy as np
import torch
from torch import optim

from config.common import NUMBER_OF_CARDS_PER_PLAYER
from config.rl import ALPHA, EPSILON_EXPLORATION_RATE_CARD_PLAY, GAMMA, EPSILON_EXPLORATION_RATE_PREDICTIONS
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import GenericFeatures
from wizard.rl_pipeline.features.select_learning_features_and_cast_to_tensor import (
    SelectLearningFeaturesAndCastToTensor,
)
from wizard.rl_pipeline.models.multi_step_ann import MultiStepANN
from wizard.rl_pipeline.type import Action, ActionType


class DQNAgent:
    NUMBER_GRAD_ACCUMULATION_STEPS = 100

    def __init__(self, model: MultiStepANN):
        self.model = model
        self._optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        self._deterministic_behavior = False
        self._n_iter = 0

    def train(
        self,
        state: GenericFeatures,
        action: Action,
        next_state: GenericFeatures,
        reward: int,
    ):
        state = self.update_state_to_action_state_for_prediction_phase_only(state, action)
        q_state = self._get_q_state(state, action)
        q_next_state = self._get_q_next_state(next_state)

        loss = (GAMMA * q_next_state + reward - q_state).pow(2).mean() / self.NUMBER_GRAD_ACCUMULATION_STEPS
        loss.backward()
        self._n_iter += 1

        if self._n_iter % self.NUMBER_GRAD_ACCUMULATION_STEPS == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        return (loss * self.NUMBER_GRAD_ACCUMULATION_STEPS).pow(0.5).item()

    @staticmethod
    def update_state_to_action_state_for_prediction_phase_only(state: GenericFeatures, action: Action) -> GenericFeatures:
        if action.is_prediction:
            state = deepcopy(state)
            state.generic_objective_context.NUMBER_ROUNDS_TO_WIN = action.value
        return state

    def select_action(self, state: GenericFeatures) -> Action:
        if state.is_prediction_step:
            return Action(ActionType.Prediction, self.select_prediction(state))
        return Action(ActionType.CardToPlay, self.select_card_to_play(state))

    def select_prediction(self, state: GenericFeatures) -> int:
        predictions_sorted_by_priority = self._get_predictions_sorted_by_priority(state)
        return (
            predictions_sorted_by_priority[0]
            if predictions_sorted_by_priority[0] != state.generic_objective_context.FORBIDDEN_PREDICTION
            else predictions_sorted_by_priority[1]
        )

    def select_card_to_play(self, state: GenericFeatures) -> str:
        with torch.no_grad():
            state_feat_torch = SelectLearningFeaturesAndCastToTensor().execute(state)
            action, _ = self._select_card_play_eps_greedy_action(self.model.forward(*state_feat_torch))
            return Card.from_id(action.item()).representation

    def q_max(self, state: GenericFeatures) -> float:
        with torch.no_grad():
            state_feat_torch = SelectLearningFeaturesAndCastToTensor().execute(state)
            _, q_value = self._select_card_play_eps_greedy_action(
                self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
            )
            return q_value.item()

    def set_deterministic_behavior(self, deterministic: bool) -> None:
        self._deterministic_behavior = deterministic

    def _get_q_state(self, state: GenericFeatures, action: Action) -> torch.Tensor:
        state_feat_torch = SelectLearningFeaturesAndCastToTensor().execute(state)

        if action.is_card_to_play:
            return self.model.forward(*state_feat_torch)[Card.from_representation(action.value).id]
        return self._select_card_play_eps_greedy_action(
            self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
        )[1]

    def _get_q_next_state(self, next_state: GenericFeatures) -> torch.Tensor:
        with torch.no_grad():
            if not next_state.generic_objective_context.IS_TERMINAL:
                next_state_feat_torch = SelectLearningFeaturesAndCastToTensor().execute(next_state)
                return self._select_card_play_eps_greedy_action(
                    self.model.forward(*next_state_feat_torch), epsilon_exploration_rate=0
                )[1]
            return torch.tensor(0, dtype=torch.float32)

    def _get_predictions_sorted_by_priority(self, state: GenericFeatures) -> list[int]:
        q_values = []
        if not self._deterministic_behavior and random.random() < EPSILON_EXPLORATION_RATE_PREDICTIONS:
            all_actions = list(range(NUMBER_OF_CARDS_PER_PLAYER + 1))
            random.shuffle(all_actions)
            return all_actions
        with torch.no_grad():
            for action in range(NUMBER_OF_CARDS_PER_PLAYER + 1):
                state.generic_objective_context.NUMBER_ROUNDS_TO_WIN = action
                state_feat_torch = SelectLearningFeaturesAndCastToTensor().execute(state)
                _, q_value = self._select_card_play_eps_greedy_action(
                    self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
                )
                q_values.append((action, q_value.item()))
        return [action for action, _ in sorted(q_values, key=lambda ele: ele[1], reverse=True)]

    def _select_card_play_eps_greedy_action(
        self,
        q_for_playable_cards: torch.Tensor,
        epsilon_exploration_rate: float = EPSILON_EXPLORATION_RATE_CARD_PLAY,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        non_masked_indices = torch.nonzero(q_for_playable_cards)
        if not self._deterministic_behavior and random.random() < epsilon_exploration_rate:
            selected_action = non_masked_indices[np.random.randint(0, non_masked_indices.size(0))]
        else:
            selected_action = non_masked_indices[q_for_playable_cards[non_masked_indices].argmax()]
        return selected_action, q_for_playable_cards[selected_action]

