import pandas as pd

from card import Card


class AdapterCsv:
    def __init__(self, path: str):
        self.path = path

    def base_csv_to_csv_all_combinations(self):
        pass


class AdapterCsvTwoCards(AdapterCsv):
    def __init__(self, path: str):
        super().__init__(path)

    def base_csv_to_csv_all_combinations(self):
        df = pd.read_csv(self.path)
        df_copy = df.copy().rename(columns={'first_card': 'second_card', 'second_card': 'first_card'})
        df_all_combinations = pd.concat((df, df_copy))
        return df_all_combinations


class ReadableStatTable:
    def __init__(self, stat_table: pd.DataFrame, ranking_colors: list[str] = ['RED', 'BLUE', 'GREEN', 'YELLOW']):
        self.stat_table = stat_table
        self.ranking_colors = ranking_colors

    def get_info_hand(self, cards: list[Card]):
        pass

    def _get_number_trump(self, cards: list[Card]):
        counter = 0
        for card in cards:
            if card.color == self.ranking_colors[0]:
                counter += 1
        return counter

    @staticmethod
    def _get_number_joker(cards: list[Card]):
        counter = 0
        for card in cards:
            if card.special_card is not None:
                counter += 1
        return counter


class StatTableTwoCards(ReadableStatTable):
    def __init__(self):
        super().__init__()

    def _replace_hands_color_to_get_proba(self, cards: list[Card]):
        cards_to_read_proba = cards.copy()
        number_joker = self._get_number_joker(cards)
        number_trump = self._get_number_trump(cards)
        if number_joker + number_trump == 2:
            pass
        elif (number_trump == 1) or (number_joker == 1):
            for card in cards_to_read_proba:
                if card.color != self.ranking_colors[0]:
                    card.color = self.ranking_colors[1]
        elif cards_to_read_proba[0].color == cards_to_read_proba[1].color:
            for card in cards_to_read_proba:
                card.color = self.ranking_colors[1]
        elif cards_to_read_proba[0].number >= cards_to_read_proba[1].number:
            cards_to_read_proba[0].color = self.ranking_colors[1]
            cards_to_read_proba[1].color = self.ranking_colors[2]
        else:
            cards_to_read_proba[0].color = self.ranking_colors[2]
            cards_to_read_proba[1].color = self.ranking_colors[1]
        return cards_to_read_proba

    @staticmethod
    def hand_to_readable_string(cards: list[Card]):
        list_readable_string = []
        for card in cards:
            if card.special_card is not None:
                list_readable_string += [f"{card.special_card}"]
            else:
                list_readable_string += [f"{card.number} {card.color}"]
        return list_readable_string

    def get_info_hand(self, cards: list[Card]):
        cards_to_read_proba = self._replace_hands_color_to_get_proba(cards)
        list_readable_string = self.hand_to_readable_string(cards_to_read_proba)
        df_info = self.stat_table[(self.stat_table["first_card"] == list_readable_string[0]) & (
                    self.stat_table["second_card"] == list_readable_string[1])]
        return df_info
