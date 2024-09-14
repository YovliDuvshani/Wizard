import random

import numpy as np
import torch
from torch import optim, nn

from config.common import NUMBER_CARDS_PER_PLAYER
from config.rl import ALPHA, EPSILON_EXPLORATION_RATE, GAMMA
from wizard.rl_pipeline.models.ann_pipeline import ANNPipeline


class DQNAgent:
    def __init__(self, model: ANNPipeline):
        self._model = model
        self._optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        self._deterministic_action_choice = False
        self._loss = torch.tensor(0, dtype=float)
        self.n_iter = 0

    def train(
        self,
        state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ],
        next_state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ],
        reward: int,
    ):
        state_feat_torch = self._convert_np_to_features(state_feat)
        next_state_feat_torch = self._convert_np_to_features(next_state_feat)

        _, q_state = self._select_eps_greedy_action(self._model.forward(*state_feat_torch))
        if next_state_feat[0] != {}:
            _, q_next_state = self._select_eps_greedy_action(
                self._model.forward(*next_state_feat_torch), epsilon_exploration_rate=0
            )
        q_next_state = 0

        self._loss += (GAMMA * q_next_state + reward - q_state).pow(2).mean()  # TODO: Replace with Huber loss
        self.n_iter += 1
        if self.n_iter % 100 == 99:
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1)
            self._loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._loss = torch.tensor(0, dtype=float)

    def select_action(
        self,
        state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ],
    ):
        with torch.no_grad():
            state_feat_torch = self._convert_np_to_features(state_feat)
            if self._deterministic_action_choice:
                eps_exploration_rate = 0
            else:
                eps_exploration_rate = EPSILON_EXPLORATION_RATE
            action, _ = self._select_eps_greedy_action(self._model.forward(*state_feat_torch), eps_exploration_rate)
            return action

    def get_highest_rewards_predictions(
        self,
        state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ],
    ):
        q_values = []
        with torch.no_grad():
            for action in range(NUMBER_CARDS_PER_PLAYER + 1):
                state_feat_torch = self._convert_np_to_features(state_feat)
                state_feat_torch[2][0] = action
                _, q_value = self._select_eps_greedy_action(self._model.forward(*state_feat_torch))
                q_values.append((action, q_value.item()))
        return [action for action, _ in sorted(q_values, key=lambda ele: ele[1])[:2]]

    @staticmethod
    def _select_eps_greedy_action(
        q_per_card: dict[int, torch.Tensor],
        epsilon_exploration_rate: float = EPSILON_EXPLORATION_RATE,
    ):
        q_per_card_np = {card_id: q.item() for card_id, q in q_per_card.items()}
        if random.random() < epsilon_exploration_rate:
            selected_action = random.choice(list(q_per_card.keys()))
            return selected_action, q_per_card[selected_action]
        max_action = max(q_per_card_np, key=q_per_card_np.get)
        return max_action, q_per_card[max_action]

    @staticmethod
    def _convert_np_to_features(
        state_feat: tuple[
            dict[str, np.ndarray],
            np.ndarray,
            np.ndarray,
        ]
    ):
        return (
            {
                card: torch.tensor(card_feat, requires_grad=True, dtype=float)
                for card, card_feat in state_feat[0].items()
            },
            torch.tensor(state_feat[1], requires_grad=True, dtype=float),
            torch.tensor(state_feat[2], requires_grad=True, dtype=float),
        )

    def set_deterministic_action_choice(self, deterministic: bool):
        self._deterministic_action_choice = deterministic
