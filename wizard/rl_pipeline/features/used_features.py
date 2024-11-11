from gymnasium.spaces import Discrete

from config.common import (
    NUMBER_CARDS_PER_COLOR,
    NUMBER_OF_CARDS_PER_PLAYER,
    NUMBER_OF_COLORS,
    NUMBER_OF_PLAYERS,
)
from wizard.rl_pipeline.features.data_cls import (
    FeatureDescription,
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericObjectiveContextFeatures,
)

USED_FEATURES = [
    FeatureDescription("IS_PLAYABLE", Discrete(2), group=GenericCardSpecificFeatures),
    FeatureDescription("IS_TRUMP", Discrete(2), group=GenericCardSpecificFeatures),
    FeatureDescription("IS_MAGICIAN", Discrete(2), group=GenericCardSpecificFeatures),
    FeatureDescription("IS_JESTER", Discrete(2), group=GenericCardSpecificFeatures),
    FeatureDescription("COLOR", Discrete(NUMBER_OF_COLORS + 1), group=GenericCardSpecificFeatures),
    FeatureDescription(
        "NUMBER",
        Discrete(NUMBER_CARDS_PER_COLOR + 1),
        group=GenericCardSpecificFeatures,
    ),
    FeatureDescription("CAN_WIN_CURRENT_SUB_ROUND", Discrete(2), group=GenericCardSpecificFeatures),
    FeatureDescription("WILL_WIN_CURRENT_SUB_ROUND", Discrete(2), group=GenericCardSpecificFeatures),
    # FeatureDescription(
    #     "NUMBER_SUPERIOR_CARDS_REMAINING_SAME_COLOR",
    #     Discrete(NUMBER_CARDS_PER_COLOR + 1),
    #     group=GenericCardSpecificFeatures,
    # ),
    # FeatureDescription(
    #     "NUMBER_SUPERIOR_CARDS_REMAINING_AMONG_SPECIAL_TRUMP_AND_SAME_COLOR",
    #     Discrete(NUMBER_CARDS_PER_COLOR * 2 + 1),
    #     group=GenericCardSpecificFeatures,
    # ),
    # FeatureDescription(
    #     "NUMBER_CARDS_REMAINING_IN_PLAYER_HAND",
    #     Discrete(NUMBER_OF_CARDS_PER_PLAYER + 1),
    #     group=GenericCardsContextFeatures,
    # ),
    FeatureDescription(
        "NUMBER_ROUNDS_TO_WIN",
        Discrete(NUMBER_OF_CARDS_PER_PLAYER + 1),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "NUMBER_ROUNDS_ALREADY_WON",
        Discrete(NUMBER_OF_CARDS_PER_PLAYER + 1),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "IS_PLAYER_STARTING",
        Discrete(2),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "PLAYER_POSITION",
        Discrete(NUMBER_OF_PLAYERS),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "IS_TERMINAL",
        Discrete(2),
        group=GenericObjectiveContextFeatures,
    ),
    FeatureDescription(
        "IS_PREDICTION_STEP",
        Discrete(2),
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
