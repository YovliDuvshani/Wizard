from dataclasses import dataclass, fields

import numpy as np

from wizard.rl_pipeline.features.data_cls import (
    GenericFeatures,
    GenericCardSpecificFeatures,
    GenericCardsContextFeatures,
    GenericObjectiveContextFeatures,
    FeatureGroupTypes,
    FeatureGroups,
)
from wizard.rl_pipeline.features.used_features import USED_FEATURES


class SelectLearningFeatures:
    def execute(self, generic_features: GenericFeatures) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        card_features = {
            card: self._select_and_cast_features_of_given_group(
                generic_card_specific_features, GenericCardSpecificFeatures
            )
            for card, generic_card_specific_features in generic_features.generic_card_specific.items()
        }
        cards_context_features = self._select_and_cast_features_of_given_group(
            generic_features.generic_cards_context, GenericCardsContextFeatures
        )
        objective_context_features = self._select_and_cast_features_of_given_group(
            generic_features.generic_objective_context, GenericObjectiveContextFeatures
        )
        return card_features, cards_context_features, objective_context_features

    def _select_and_cast_features_of_given_group(
        self, features: FeatureGroups, feature_group: FeatureGroupTypes
    ) -> np.ndarray:
        return np.array(
            self._convert_dataclass_to_float_tuple(
                datacls_instance=features,
                attr_subset=[feat.name for feat in USED_FEATURES if feat.group == feature_group],
            )
        )

    @staticmethod
    def _convert_dataclass_to_float_tuple(datacls_instance: dataclass, attr_subset: list[str]):
        return tuple(
            float(getattr(datacls_instance, field.name))
            for field in fields(datacls_instance)
            if field.name in attr_subset
        )
