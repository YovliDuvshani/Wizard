import random

import numpy as np
import torch
from torch import optim

from config.common import NUMBER_OF_CARDS_PER_PLAYER
from config.rl import ALPHA, EPSILON_EXPLORATION_RATE, GAMMA
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import GenericFeatures
from wizard.rl_pipeline.features.select_learning_features import SelectLearningFeatures
from wizard.rl_pipeline.models.multi_step_ann import MultiStepANN


class DQNAgent:
    NUMBER_GRAD_ACCUMULATION_STEPS = 100

    def __init__(self, model: MultiStepANN):
        self.model = model
        self._optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        self._deterministic_action_choice = False
        self._n_iter = 0

    def train(
        self,
        state: GenericFeatures,
        next_state: GenericFeatures,
        reward: int,
    ):
        state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
        next_state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(next_state))

        _, q_state = self._select_eps_greedy_action(self.model.forward(*state_feat_torch))
        if not next_state.generic_objective_context.IS_TERMINAL:
            _, q_next_state = self._select_eps_greedy_action(
                self.model.forward(*next_state_feat_torch), epsilon_exploration_rate=0
            )
        else:
            q_next_state = 0

        loss = (GAMMA * q_next_state + reward - q_state).pow(2).mean()
        loss.backward()
        self._n_iter += 1

        if self._n_iter % self.NUMBER_GRAD_ACCUMULATION_STEPS == 0:  # TODO: Move to dedicated memory class
            # nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1)
            self._optimizer.step()
            self._optimizer.zero_grad()

        return loss.pow(0.5).item()

    def select_action(self, state_feat: GenericFeatures):
        with torch.no_grad():
            state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state_feat))
            if self._deterministic_action_choice:
                eps_exploration_rate = 0
            else:
                eps_exploration_rate = EPSILON_EXPLORATION_RATE
            action, _ = self._select_eps_greedy_action(self.model.forward(*state_feat_torch), eps_exploration_rate)
            return Card.from_id(action.item()).representation

    def get_highest_rewards_predictions(self, state: GenericFeatures):
        q_values = []
        with torch.no_grad():
            for action in range(NUMBER_OF_CARDS_PER_PLAYER + 1):
                state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
                state_feat_torch[2][0] = action
                _, q_value = self._select_eps_greedy_action(self.model.forward(*state_feat_torch))
                q_values.append((action, q_value.item()))
        return [action for action, _ in sorted(q_values, key=lambda ele: ele[1])[:2]]

    def q_max(self, state: GenericFeatures):
        with torch.no_grad():
            state_feat_torch = self.convert_array_to_tensor(SelectLearningFeatures().execute(state))
            _, q_value = self._select_eps_greedy_action(
                self.model.forward(*state_feat_torch), epsilon_exploration_rate=0
            )
            return q_value.item()

    @staticmethod
    def _select_eps_greedy_action(
        q_for_playable_cards: torch.Tensor,
        epsilon_exploration_rate: float = EPSILON_EXPLORATION_RATE,
    ):
        non_masked_indices = torch.nonzero(q_for_playable_cards)
        if random.random() < epsilon_exploration_rate:
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

    def set_deterministic_action_choice(self, deterministic: bool):
        self._deterministic_action_choice = deterministic
