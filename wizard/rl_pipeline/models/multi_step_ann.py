from dataclasses import asdict
from typing import Dict

import torch

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import (
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericObjectiveContextFeatures,
)
from wizard.rl_pipeline.features.used_features import NUMBER_FEATURES_PER_GROUP
from wizard.rl_pipeline.models.utils import ANNSpecification, create_ann


class MultiStepANN(torch.nn.Module):
    def __init__(
        self,
        card_ann_specification: ANNSpecification,
        hand_ann_specification: ANNSpecification,
        strategy_ann_specification: ANNSpecification,
        q_ann_specification: ANNSpecification,
    ):
        super().__init__()
        self._card_ann = create_ann(
            input_size=NUMBER_FEATURES_PER_GROUP[GenericCardSpecificFeatures.__name__], **asdict(card_ann_specification)
        )
        self._hand_ann = create_ann(
            input_size=card_ann_specification.output_size * NUMBER_OF_UNIQUE_CARDS
            + NUMBER_FEATURES_PER_GROUP[GenericCardsContextFeatures.__name__],
            **asdict(hand_ann_specification)
        )
        self._strategy_ann = create_ann(
            input_size=hand_ann_specification.output_size
            + NUMBER_FEATURES_PER_GROUP[GenericObjectiveContextFeatures.__name__],
            **asdict(strategy_ann_specification)
        )
        self._q_ann = create_ann(
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
        playable_cards_id = [
            Card.from_representation(card_representation).id
            for card_representation, card_features in cards_features.items()
            if card_features[0].item() == 1.0
        ]  # Short-term solution
        card_embeddings = {
            Card.from_representation(card_representation).id: self._card_ann(card_features)
            for card_representation, card_features in cards_features.items()
        }

        concatenated_cards_embeddings = self._get_concatenated_cards_embeddings(card_embeddings)
        hand_ann_input_features = torch.concat((concatenated_cards_embeddings, hand_features))
        hand_representation = self._hand_ann(hand_ann_input_features)

        strategy_ann_input_features = torch.concat((hand_representation, strategy_features))
        strategy_representation = self._strategy_ann(strategy_ann_input_features)

        q_per_card = torch.zeros(NUMBER_OF_UNIQUE_CARDS, dtype=torch.float32) # What happens with Q=0? -> Masking should be handled more elegantly
        for card_id in playable_cards_id:
            q_ann_input_features = torch.concat((strategy_representation, card_embeddings[card_id]))
            q_per_card[card_id] = self._q_ann(q_ann_input_features)

        return q_per_card

    def _get_concatenated_cards_embeddings(self, cards_embeddings: Dict[int, torch.Tensor]):
        card_ids = list(set(cards_embeddings.keys()))
        card_ids.sort()
        result = torch.Tensor()
        for card_id in card_ids:
            masked_cards = torch.zeros(self._card_ann_output_size * card_id - result.shape[0])
            result = torch.concat((result, masked_cards, cards_embeddings[card_id]))
        result = torch.concat(
            (
                result,
                torch.zeros(self._card_ann_output_size * NUMBER_OF_UNIQUE_CARDS - result.shape[0]),
            )
        )
        return result
