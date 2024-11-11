import torch

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.base_game.card import Card
from wizard.rl_pipeline.features.data_cls import GenericCardsContextFeatures, GenericObjectiveContextFeatures
from wizard.rl_pipeline.features.used_features import NUMBER_FEATURES_PER_GROUP
from wizard.rl_pipeline.models.utils import ANNSpecification, create_ann


class NotSupportedNumberOfCards(Exception):
    pass


class BaseANN(torch.nn.Module):
    def __init__(self, ann_specification: ANNSpecification):
        super().__init__()
        self._ann = create_ann(
            input_size=NUMBER_FEATURES_PER_GROUP[GenericCardsContextFeatures.__name__]
            + NUMBER_FEATURES_PER_GROUP[GenericObjectiveContextFeatures.__name__],  # NUMBER_OF_UNIQUE_CARDS
            hidden_layers_size=ann_specification.hidden_layers_size,
            output_size=1,
        )

    def forward(
        self,
        cards_features: dict[str, torch.Tensor],
        hand_features: torch.Tensor,
        strategy_features: torch.Tensor,
    ):
        if len(list(cards_features.keys())) > 1:
            raise NotSupportedNumberOfCards
        card_representation, _ = next(iter(cards_features)), next(iter(cards_features.values()))
        ohe_card = torch.zeros(NUMBER_OF_UNIQUE_CARDS, dtype=torch.float32)
        ohe_card[Card.from_representation(card_representation).id] = 1

        input_tensor = torch.concat(
            (hand_features, strategy_features)
        )  # torch.concat((ohe_card, hand_features, strategy_features))

        output_tensor = torch.zeros(NUMBER_OF_UNIQUE_CARDS, dtype=torch.float32)
        output_tensor[Card.from_representation(card_representation).id] = self._ann(input_tensor)
        return output_tensor
