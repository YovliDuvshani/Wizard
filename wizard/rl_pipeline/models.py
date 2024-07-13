from dataclasses import dataclass
from typing import List, Any, Dict

import torch

from config.common import NUMBER_OF_UNIQUE_CARDS

ANN_BASE_HIDDEN_LAYERS_SIZE = {
    "CARD_REPRESENTATION_ANN": [5, 5],
    "HAND_REPRESENTATION_ANN": [5, 5],
    "STRATEGY_REPRESENTATION_ANN": [5, 5],
}
CARD_FEATURES_SIZE = 5
CARD_REPRESENTATION_SIZE = 5


@dataclass
class ANNDefinition:
    input_size: int
    hidden_layers_size: List[int]
    output_size: int


class ANNPipeline(torch.nn.Module):
    def __init__(
        self,
        card_ann_definition: ANNDefinition,
        hand_ann_definition: ANNDefinition,
        strategy_ann_definition: ANNDefinition,
        q_ann_definition: ANNDefinition,
    ):
        super().__init__()
        self._card_ann = self._create_ann(card_ann_definition)
        self._hand_ann = self._create_ann(hand_ann_definition)
        self._strategy_ann = self._create_ann(strategy_ann_definition)
        self._q_ann = self._create_ann(q_ann_definition)

    def forward(
        self,
        cards_features: Dict[int, Any],
        hand_features: torch.Tensor,
        strategy_features: torch.Tensor,
        playable_cards_id: [List[int]],
    ):
        card_representations = {
            card_id: self._card_ann(card_features)
            for card_id, card_features in cards_features.items()
        }

        ohe_card_representations = self._get_ohe_card_representation(
            card_representations
        )
        hand_ann_input_features = torch.concat(
            (ohe_card_representations, hand_features)
        )
        hand_representation = self._hand_ann(hand_ann_input_features)

        strategy_ann_input_features = torch.concat(
            (hand_representation, strategy_features)
        )
        strategy_representation = self._strategy_ann(strategy_ann_input_features)

        q_per_card = {}
        for card_id in playable_cards_id:
            q_ann_input_features = torch.concat(
                (strategy_representation, card_representations[card_id])
            )
            q_per_card[card_id] = self._q_ann(q_ann_input_features)

        return q_per_card

    @staticmethod
    def _create_ann(ann_definition: ANNDefinition):
        layer_sizes = [
            ann_definition.input_size,
            *ann_definition.hidden_layers_size,
            ann_definition.output_size,
        ]
        ann = []
        for ind in range(len(layer_sizes) - 2):
            ann.append(torch.nn.Linear(layer_sizes[ind], layer_sizes[ind + 1]))
            ann.append(torch.nn.ReLU())
        ann.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        return torch.nn.Sequential(*ann)

    @staticmethod
    def _get_ohe_card_representation(cards_representations: Dict[int, torch.Tensor]):
        card_ids = list(set(cards_representations.keys()))
        card_ids.sort()
        result = torch.Tensor()
        for card_id in card_ids:
            masked_cards = torch.zeros(CARD_REPRESENTATION_SIZE * card_id - len(result))
            result = torch.concat((result, masked_cards, cards_representations[card_id]))
        result = torch.concat(
            (
                result,
                torch.zeros(
                    CARD_REPRESENTATION_SIZE * NUMBER_OF_UNIQUE_CARDS - len(result)
                ),
            )
        )
        return result
