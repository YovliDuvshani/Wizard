class Card:
    def __init__(self, color: str, number: int, special_card: str):
        self.special_card = special_card
        self.number = number
        self.color = color

    def is_equal(self, card):
        if ((self.number == card.number) & (self.color == card.color)) or ((self.special_card == card.special_card) & (
                self.special_card is not None)):
            return True
        return False

    def is_in_list(self, list_cards):
        for card in list_cards:
            if self.is_equal(card):
                return True
        return False

    def get_number(self):
        return self.number

    def get_color(self):
        return self.color

    def get_special_card(self):
        return self.special_card

    def prepare_card_for_display(self):
        if self.color is not None:
            return f'{self.number} {self.color}'
        else:
            return str(self.special_card)
