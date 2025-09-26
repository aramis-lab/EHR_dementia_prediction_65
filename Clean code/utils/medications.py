import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple, Dict

DataFrameStatsTuple = Tuple[pd.DataFrame, Dict[str, int]]

get_year = lambda x: int(x[:4])

def get_one_hot_encoded_medications(df, n_most_frequent:int, start_year:int, end_year:int)->DataFrameStatsTuple:
    """
    Returns a pandas dataframe of the one hot encoded `n_most_frequent` most frequent
    medications taken between `start_year` and `end_year` (inclusive).
    
    Args:
        - df (pd.DataFrame): A pandas dataframe with the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `contact_date` (str): The date when the medication was taken (in yyyy-mm-dd format)
            * `product_atc_code` (str): The ATC code of the medication
        - n_most_frequent (int or None): The number of most frequent medications to return. If None, all will be returned.
        - start_year (int): The first year to consider when selecting medications.
        - end_year (int): The last year to consider when selecting medications.

    Returns:
        - A pandas dataframe with person_id and one hot encoded columns of the taken medications (one column for 
        each medication, indicating whether the person has taken that medication during the considered time window).
        - A dictionary containing additional statistics:
            * `n_patients_med` (int): The number of unique patients in the input dataframe.
    """
    
    #### Pandas df: `person_id`:int, `medications`:List[str]
    n_patients_med = df['person_id'].nunique()
    print(f'\t[INFO] {n_patients_med:,} unique patients')
    print('\tExtracting medications...')
    df['contact_year'] = df.pop('contact_date').apply(get_year)
    group = df.groupby(by='person_id')
    
    def helper(x):
        mask = (x['contact_year'] < start_year) | (x['contact_year'] > end_year) | (x['product_atc_code'].isna())
        return list({e[:3] for e in x['product_atc_code'][~mask]})
    
    medications = group.apply(helper)
    medications = pd.DataFrame(medications, columns=['medications'])
    medications.reset_index(inplace=True)
    
    ### OneHotEncoding of medications
    # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
    print('\tOneHotEncoding them...')
    mlb = MultiLabelBinarizer(sparse_output=True)
    medications = medications.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(medications.pop('medications')),
            index=medications.index,
            columns=mlb.classes_
        ).astype(int)
    )
    
    # take the most frequent medications
    if n_most_frequent:
        assert n_most_frequent > 0
        # offset the start by one column to ignore `person_id`
        print(f'\tTaking the {n_most_frequent} most frequent ones (out of the {len(medications.columns)-1})...')
        most_freq_meds = list(medications.iloc[:,1:].sum(numeric_only=True).nlargest(n=n_most_frequent).index)
        medications = medications[['person_id']+most_freq_meds]
        
    return medications, {'n_patients_med':n_patients_med}