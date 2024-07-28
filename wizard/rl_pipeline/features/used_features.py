from gymnasium.spaces import Discrete

from config.common import (
    NUMBER_CARDS_PER_COLOR,
    NUMBER_CARDS_PER_PLAYER,
    NUMBER_OF_COLORS,
)
from wizard.rl_pipeline.features.data_cls import (
    FeatureDescription,
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericObjectiveContextFeatures,
)

USED_FEATURES = [
    FeatureDescription("IS_TRUMP", Discrete(1), group=GenericCardSpecificFeatures),
    FeatureDescription("IS_MAGICIAN", Discrete(1), group=GenericCardSpecificFeatures),
    FeatureDescription("IS_JESTER", Discrete(1), group=GenericCardSpecificFeatures),
    FeatureDescription("COLOR", Discrete(NUMBER_OF_COLORS + 1), group=GenericCardSpecificFeatures),
    FeatureDescription(
        "NUMBER",
        Discrete(NUMBER_CARDS_PER_COLOR + 1),
        group=GenericCardSpecificFeatures,
    ),
    FeatureDescription("IS_PLAYABLE", Discrete(1), group=GenericCardSpecificFeatures),
    FeatureDescription(
        "NUMBER_CARDS_REMAINING_IN_PLAYER_HAND",
        Discrete(NUMBER_CARDS_PER_PLAYER + 1),
        group=GenericCardsContextFeatures,
    ),
    FeatureDescription(
        "NUMBER_ROUNDS_TO_WIN",
        Discrete(NUMBER_CARDS_PER_PLAYER + 1),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "NUMBER_ROUNDS_ALREADY_WON",
        Discrete(NUMBER_CARDS_PER_PLAYER + 1),
        group=GenericObjectiveContextFeatures,
    ),
]

NUMBER_FEATURES_PER_GROUP = {
    group.__name__: len([feat for feat in USED_FEATURES if feat.group == group])
    for group in [
        GenericCardSpecificFeatures,
        GenericCardsContextFeatures,
        GenericObjectiveContextFeatures,
    ]
}
