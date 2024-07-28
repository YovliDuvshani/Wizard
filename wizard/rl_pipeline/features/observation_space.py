from gymnasium.spaces import Discrete, Sequence, Tuple

from config.common import NUMBER_OF_UNIQUE_CARDS
from wizard.rl_pipeline.features.data_cls import (
    GenericCardsContextFeatures,
    GenericCardSpecificFeatures,
    GenericObjectiveContextFeatures,
)
from wizard.rl_pipeline.features.used_features import USED_FEATURES

OBSERVATION_SPACE = Tuple(
    (
        Sequence(
            Tuple(
                (
                    Discrete(NUMBER_OF_UNIQUE_CARDS),
                    Tuple([feat.space for feat in USED_FEATURES if feat.group == GenericCardSpecificFeatures]),
                ),
            )
        ),
        Tuple([feat.space for feat in USED_FEATURES if feat.group == GenericCardsContextFeatures]),
        Tuple([feat.space for feat in USED_FEATURES if feat.group == GenericObjectiveContextFeatures]),
    )
)
