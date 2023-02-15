import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('2_cards_everybody_against_you')
    df['best_prediction'] = df[
        ['best_outcome_prediction_0', 'best_outcome_prediction_1', 'best_outcome_prediction_2']].values.argmax(1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 1000)
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'first_card': 'second_card', 'second_card': 'first_card'})
    df = pd.concat((df,df_copy))
    df = df.drop(columns='Unnamed: 0')
    print(df.to_dict('tight'))
