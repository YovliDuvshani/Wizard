import torch

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.rl_pipeline.models.ann_pipeline import ANNDefinition, ANNPipeline

ann = ANNPipeline(
    card_ann_definition=ANNDefinition(3, [3], 5),
    hand_ann_definition=ANNDefinition(5 * NUMBER_OF_UNIQUE_CARDS + 3, [3], 5),
    strategy_ann_definition=ANNDefinition(5, [], 5),
    q_ann_definition=ANNDefinition(10, [3, 4], 1),
)

ann.forward(
    cards_features={
        3: torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
        8: torch.tensor([1.0, 1.0, 1.0], requires_grad=True),
    },
    hand_features=torch.tensor([1.0, 1.0, 1.0], requires_grad=True),
    strategy_features=torch.tensor([]),
    playable_cards_id=[3, 8],
)
