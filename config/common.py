NUMBER_OF_PLAYERS = 3
NUMBER_OF_CARDS_PER_PLAYER = 1

NUMBER_CARDS_PER_COLOR = 13
NUMBER_OF_JESTERS = 2
NUMBER_OF_MAGICIANS = 2
JESTER_NAME = "Jester"
MAGICIAN_NAME = "Magician"
BASE_COLORS = ["RED", "BLUE", "GREEN", "YELLOW"]
TRUMP_COLOR = BASE_COLORS[0]
SUITS = list(range(1, NUMBER_CARDS_PER_COLOR + 1))

NUMBER_OF_COLORS = 4
NUMBER_OF_CARDS = len(SUITS) * len(BASE_COLORS) + NUMBER_OF_JESTERS + NUMBER_OF_MAGICIANS
NUMBER_OF_UNIQUE_CARDS = len(SUITS) * len(BASE_COLORS) + 2  # TODO: Handle edge case where no magician/jester

BASE_REWARD = 20
DYNAMIC_REWARD = 10
DYNAMIC_LOSS = -10
