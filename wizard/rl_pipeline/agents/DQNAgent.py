import random
from copy import deepcopy

import numpy as np
import torch
from torch import optim

from config.common import NUMBER_OF_CARDS_PER_PLAYER
from config.rl import ALPHA, EPSILON_EXPLORATION_RATE_CARD_PLAY, GAMMA, EPSILON_EXPLORATION_RATE_PREDICTIONS
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import GenericFeatures
from wizard.rl_pipeline.features.select_learning_features import SelectLearningFeatures
from wizard.rl_pipeline.models.multi_step_ann import MultiStepANN


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
        action: int | str,
        next_state: GenericFeatures,
        reward: int,
    ):
        state = self.update_state_to_action_state_for_prediction_phase(state, action)

        state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
        next_state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(next_state))

        if isinstance(action, str):
            q_state = self.model.forward(*state_feat_torch)[Card.from_representation(action).id]
        elif isinstance(action, int):
            _, q_state = self._select_eps_greedy_action(
                self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
            )

        with torch.no_grad():
            if not next_state.generic_objective_context.IS_TERMINAL:
                _, q_next_state = self._select_eps_greedy_action(
                    self.model.forward(*next_state_feat_torch), epsilon_exploration_rate=0
                )
            else:
                q_next_state = torch.tensor(0, dtype=torch.float32)

        loss = (GAMMA * q_next_state + reward - q_state).pow(2).mean() / self.NUMBER_GRAD_ACCUMULATION_STEPS
        loss.backward()
        self._n_iter += 1

        if self._n_iter % self.NUMBER_GRAD_ACCUMULATION_STEPS == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        return (loss * self.NUMBER_GRAD_ACCUMULATION_STEPS).pow(0.5).item()

    @staticmethod
    def update_state_to_action_state_for_prediction_phase(state: GenericFeatures, action: int | str):
        if isinstance(action, int):
            state = deepcopy(state)
            state.generic_objective_context.NUMBER_ROUNDS_TO_WIN = action
        return state

    def select_action(self, state: GenericFeatures) -> str | int:
        if state.generic_objective_context.IS_PREDICTION_STEP == 1:
            return (
                self.get_predictions_sorted_by_q(state)[0]
                if self.get_predictions_sorted_by_q(state)[0] != state.generic_objective_context.FORBIDDEN_PREDICTION
                else self.get_predictions_sorted_by_q(state)[1]
            )
        with torch.no_grad():
            state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
            action, _ = self._select_eps_greedy_action(self.model.forward(*state_feat_torch))
            return Card.from_id(action.item()).representation

    def get_predictions_sorted_by_q(self, state: GenericFeatures) -> list[int]:
        q_values = []
        if not self._deterministic_behavior and random.random() < EPSILON_EXPLORATION_RATE_PREDICTIONS:
            all_actions = list(range(NUMBER_OF_CARDS_PER_PLAYER + 1))
            random.shuffle(all_actions)
            return all_actions
        with torch.no_grad():
            for action in range(NUMBER_OF_CARDS_PER_PLAYER + 1):
                state.generic_objective_context.NUMBER_ROUNDS_TO_WIN = action
                state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
                _, q_value = self._select_eps_greedy_action(
                    self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
                )
                q_values.append((action, q_value.item()))
        return [action for action, _ in sorted(q_values, key=lambda ele: ele[1], reverse=True)[:2]]

    def q_max(self, state: GenericFeatures):
        with torch.no_grad():
            state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
            _, q_value = self._select_eps_greedy_action(
                self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
            )
            return q_value.item()

    def _select_eps_greedy_action(
        self,
        q_for_playable_cards: torch.Tensor,
        epsilon_exploration_rate: float = EPSILON_EXPLORATION_RATE_CARD_PLAY,
    ):
        non_masked_indices = torch.nonzero(q_for_playable_cards)
        if not self._deterministic_behavior and random.random() < epsilon_exploration_rate:
            selected_action = non_masked_indices[np.random.randint(0, non_masked_indices.size(0))]
        else:
            selected_action = non_masked_indices[q_for_playable_cards[non_masked_indices].argmax()]
        return selected_action, q_for_playable_cards[selected_action]

    @staticmethod
    def convert_array_to_tensor(
        state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ]
    ):
        return (
            {
                card: torch.tensor(card_feat, requires_grad=True, dtype=torch.float32)
                for card, card_feat in state_feat[0].items()
            },
            torch.tensor(state_feat[1], requires_grad=True, dtype=torch.float32),
            torch.tensor(state_feat[2], requires_grad=True, dtype=torch.float32),
        )

    def set_deterministic_behavior(self, deterministic: bool):
        self._deterministic_behavior = deterministic
