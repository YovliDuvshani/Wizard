from adapters import AdapterCsvTwoCards, AdapterCsvInfoTwoCards
from card import Card

if __name__ == '__main__':
    # game_manager = GameManagerTwoCardsFirstPlayer(number_trial=100)
    # game_manager.simulate()
    """
    deck = Deck(deck_id=1)
    hand_two_cards = HandCombinationsTwoCards(trump_color=deck.get_colors[0], base_color1=deck.get_colors[1],
                                              base_color2=deck.get_colors[2])
    possibilities_for_first_player_to_test = hand_two_cards.possibilities_to_test(deck)
    print(len(possibilities_for_first_player_to_test))
    print(possibilities_for_first_player_to_test[1][0].prepare_card_for_display(),
          possibilities_for_first_player_to_test[1][1].prepare_card_for_display())
    deck = FilteredDeck(id=1, cards_to_suppress=possibilities_for_first_player_to_test[1])
    for card in deck.cards:
        if card.special_card is not None:
            print(card.prepare_card_for_display())
    print(len(deck.cards))
    """

    adapter = AdapterCsvTwoCards("2_cards_everybody_against_you")
    adapter_to_info = AdapterCsvInfoTwoCards()

    adapter_to_info.assign_table(adapter.base_csv_to_csv_all_combinations())
    print(adapter.base_csv_to_csv_all_combinations().columns)
    card1 = Card(color="RED",number=12,special_card=None)
    card2 = Card(color="RED",number=5,special_card=None)

    print(adapter_to_info.get_info_hand([card1,card2]).best_outcome_prediction_2.values[0])

