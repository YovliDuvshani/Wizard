import numpy as np
import torch

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.card import Card
from wizard.rl_pipeline.models.utils import ANNSpecification, create_ann


class BaseANN(torch.nn.Module):
    def __init__(self, ann_specification: ANNSpecification):
        super().__init__()
        self._ann = create_ann(
            input_size=NUMBER_OF_UNIQUE_CARDS,  # NUMBER_FEATURES_PER_GROUP[GenericCardSpecificFeatures.__name__],  #sum(list(NUMBER_FEATURES_PER_GROUP.values())),
            hidden_layers_size=ann_specification.hidden_layers_size,
            output_size=1,
        )

    def forward(
        self,
        cards_features: dict[str, torch.Tensor],
        hand_features: torch.Tensor,
        strategy_features: torch.Tensor,
    ):
        card_representation, card_features = next(iter(cards_features)), next(iter(cards_features.values()))
        input_tensor = torch.zeros(NUMBER_OF_UNIQUE_CARDS, dtype=torch.float32)
        input_tensor[Card.from_representation(card_representation).id] = 1

        output_tensor = torch.zeros(NUMBER_OF_UNIQUE_CARDS, dtype=torch.float32)
        output_tensor[Card.from_representation(card_representation).id] = self._ann(input_tensor)
        return output_tensor
