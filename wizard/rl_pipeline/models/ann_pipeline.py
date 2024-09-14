from dataclasses import asdict, dataclass
from typing import Dict, List

import torch

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import (
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericObjectiveContextFeatures,
)
from wizard.rl_pipeline.features.used_features import NUMBER_FEATURES_PER_GROUP


@dataclass
class ANNSpecification:
    hidden_layers_size: List[int] | None = None
    output_size: int | None = None


class ANNPipeline(torch.nn.Module):
    def __init__(
        self,
        card_ann_specification: ANNSpecification,
        hand_ann_specification: ANNSpecification,
        strategy_ann_specification: ANNSpecification,
        q_ann_specification: ANNSpecification,
    ):
        super().__init__()
        self._card_ann = self._create_ann(
            input_size=NUMBER_FEATURES_PER_GROUP[GenericCardSpecificFeatures.__name__], **asdict(card_ann_specification)
        )
        self._hand_ann = self._create_ann(
            input_size=card_ann_specification.output_size * NUMBER_OF_UNIQUE_CARDS
            + NUMBER_FEATURES_PER_GROUP[GenericCardsContextFeatures.__name__],
            **asdict(hand_ann_specification)
        )
        self._strategy_ann = self._create_ann(
            input_size=hand_ann_specification.output_size
            + NUMBER_FEATURES_PER_GROUP[GenericObjectiveContextFeatures.__name__],
            **asdict(strategy_ann_specification)
        )
        self._q_ann = self._create_ann(
            input_size=card_ann_specification.output_size + strategy_ann_specification.output_size,
            hidden_layers_size=q_ann_specification.hidden_layers_size,
            output_size=1,
        )
        self._card_ann_output_size = card_ann_specification.output_size

    def forward(
        self,
        cards_features: Dict[str, torch.Tensor],
        hand_features: torch.Tensor,
        strategy_features: torch.Tensor,
    ):
        card_representations = {
            card_id: self._card_ann(card_features) for card_id, card_features in cards_features.items()
        }

        ohe_card_representations = self._get_ohe_card_representation(card_representations)
        hand_ann_input_features = torch.concat((ohe_card_representations, hand_features))
        hand_representation = self._hand_ann(hand_ann_input_features)

        strategy_ann_input_features = torch.concat((hand_representation, strategy_features))
        strategy_representation = self._strategy_ann(strategy_ann_input_features)

        q_per_card = {}
        playable_cards_id = [
            card_id for card_id, card_features in cards_features.items() if card_features[0].item() == 1.0
        ]  # Short-term solution
        for card_id in playable_cards_id:
            q_ann_input_features = torch.concat((strategy_representation, card_representations[card_id]))
            q_per_card[card_id] = self._q_ann(q_ann_input_features)

        return q_per_card

    @staticmethod
    def _create_ann(input_size: int, hidden_layers_size: list[int] | None, output_size: int):
        if hidden_layers_size:
            layer_sizes = [input_size, *hidden_layers_size, output_size]
        else:
            layer_sizes = [input_size, output_size]
        ann = []
        for ind in range(len(layer_sizes) - 2):
            ann.append(torch.nn.Linear(layer_sizes[ind], layer_sizes[ind + 1], dtype=float))
            ann.append(torch.nn.ReLU())
        ann.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1], dtype=float))
        return torch.nn.Sequential(*ann)

    def _get_ohe_card_representation(self, cards_representations: Dict[str, torch.Tensor]):
        cards_representations_per_id = {
            Card.from_representation(card_str).id: tensor for card_str, tensor in cards_representations.items()
        }
        card_ids = list(set(cards_representations_per_id.keys()))
        card_ids.sort()
        result = torch.Tensor()
        for card_id in card_ids:
            masked_cards = torch.zeros(self._card_ann_output_size * card_id - len(result))
            result = torch.concat((result, masked_cards, cards_representations_per_id[card_id]))
        result = torch.concat(
            (
                result,
                torch.zeros(self._card_ann_output_size * NUMBER_OF_UNIQUE_CARDS - len(result)),
            )
        )
        return result
