import pandas as pd
from typing import Tuple, Dict

DataFrameStatsTuple = Tuple[pd.DataFrame, Dict[str, int]]

def remove_undefined_gender(df)->DataFrameStatsTuple:
    n = df['person_id'].nunique()
    n_f = sum(df['gender_code'] == 'F')
    n_m = sum(df['gender_code'] == 'M')
    n_undef = n-n_f-n_m
    print(f'\t[INFO] We have {n:,} unique patients:\n\t\t'
          f'- {n_f/n*100:.2f}% females,\n\t\t'
          f'- {n_m/n*100:.2f}% males,\n\t\t'
          f'- {n_undef} undefined.')
    if n_undef > 0:
        print(f'\tExcluding people with undefined genders...')
        df = df[df['gender_code'].isin({'F', 'M'})]
    return df, {'n':n, 'n_undef':n_undef}