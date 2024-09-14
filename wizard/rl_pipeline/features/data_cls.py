from dataclasses import dataclass
from typing import Dict

from gymnasium.core import ObsType


@dataclass
class GenericCardSpecificFeatures:
    IS_PLAYABLE: bool
    IS_TRUMP: bool
    IS_MAGICIAN: bool
    IS_JESTER: bool
    COLOR: int
    NUMBER: int
    CAN_WIN_CURRENT_SUB_ROUND: bool = False
    WILL_WIN_CURRENT_SUB_ROUND: bool = False
    NUMBER_CARDS_REMAINING_SAME_COLOR: int | None = None
    NUMBER_SUPERIOR_CARDS_REMAINING_SAME_COLOR: int | None = None
    NUMBER_INFERIOR_CARDS_REMAINING_SAME_COLOR: int | None = None
    NUMBER_CARDS_SAME_COLOR_IN_PLAYERS_HAND: int | None = None
    NUMBER_SUPERIOR_CARDS_SAME_COLOR_IN_PLAYERS_HAND: int | None = None
    NUMBER_INFERIOR_CARDS_SAME_COLOR_IN_PLAYERS_HAND: int | None = None
    NUMBER_CARDS_REMAINING_AMONG_SPECIAL_TRUMP_AND_SAME_COLOR: int | None = None
    NUMBER_SUPERIOR_CARDS_REMAINING_AMONG_SPECIAL_TRUMP_AND_SAME_COLOR: int | None = None
    NUMBER_INFERIOR_CARDS_REMAINING_AMONG_SPECIAL_TRUMP_AND_SAME_COLOR: int | None = None


@dataclass
class GenericCardsContextFeatures:
    NUMBER_CARDS_REMAINING: int
    NUMBER_CARDS_REMAINING_IN_PLAYER_HAND: int
    NUMBER_CARDS_REMAINING_IN_OTHER_PLAYERS_HANDS: int
    NUMBER_CARDS_REMAINING_PER_COLOR: Dict[str, int]
    NUMBER_CARDS_REMAINING_IN_HAND_PER_COLOR: Dict[str, int]


@dataclass
class GenericObjectiveContextFeatures:
    NUMBER_ROUNDS_TO_WIN: int
    NUMBER_ROUNDS_ALREADY_WON: int
    TOTAL_NUMBER_OF_ROUNDS: int


@dataclass
class GenericFeatures:
    generic_card_specific: Dict[str, GenericCardSpecificFeatures]
    generic_cards_context: GenericCardsContextFeatures
    generic_objective_context: GenericObjectiveContextFeatures


FeatureGroups = GenericCardSpecificFeatures | GenericCardsContextFeatures | GenericObjectiveContextFeatures

FeatureGroupTypes = (
    type[GenericCardSpecificFeatures] | type[GenericCardsContextFeatures] | type[GenericObjectiveContextFeatures]
)


@dataclass
class FeatureDescription:
    name: str
    space: ObsType
    group: FeatureGroupTypes
