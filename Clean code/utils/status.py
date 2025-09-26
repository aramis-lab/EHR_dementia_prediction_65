import pandas as pd
from typing import Tuple, Dict
from .medications import get_year

DataFrameStatsTuple = Tuple[pd.DataFrame, Dict[str, int]]

def get_status(df, start_year, end_year, extraction_date):
    """
    Get the status of patients

    Args:
        - df (pd.DataFrame): A pandas dataframe containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `start_change_date` (str): the date the patient started the current state  (in yyyy-mm-dd format)
            * `end_change_date` (str): the date the patient ended the current state  (in yyyy-mm-dd format)
            * `person_state_code` (str): code of the patient's current state. The possible values are:
              'A' (Active), 'I' (Inactive), 'D' (Dead), 'T' (Temporaire, specific to UK), 'P' (Probably dead, specific
              to France) and 'S' (Suspected death, specific to France)
        - start_year (int): An integer representing the start year of the period of interest.
        - end_year (int): An integer representing the end year of the period of interest.
        - extraction_date (str): A string representing the date when the data was extracted (in yyyy-mm-dd format).

    Returns:
        - A pandas dataframe contains patient status information for the time period of interest, including their 
        current status and the time to reach that status (relatively to end_year). 
        - A dictionary contains additional statistics:
            * `n_status` (int): The number of unique patients having at least one status record in the input dataframe 
            * `n_inactive_1` (int): The number of unique patients being inactive at least once during the period 
            * `n_died_be` (int): The number of unique patients who diead before end_year 
            * `n_T` (int): The number of unique patients who where always 'Temporaire' or always 'Inactive' 
            * `n_inactive_ue` (int): The number of unique patients who became either 'Temporaire' or always during the 
            period and remained in that state until the end
        - A set contains patient IDs that  were inactive at least once during the time period of interest. 
    """
    n = df['person_id'].nunique()
    stats = {'n_status':n}
    print(f'\t[INFO] {n:,} patients with at least one status record')
    
    end_date = pd.to_datetime(f'{end_year}-12-31', format='%Y-%m-%d')
    extraction_date = pd.to_datetime(extraction_date, format='%Y-%m-%d')
    
    ## 
    status_counts = df['person_id'].value_counts()
    only_one_status_ids =  status_counts[status_counts == 1].index

    # Set person_state_code as 'A' for individuals with only one status entry case where their status is always 
    # D, S, P, I, T (start_change_date='0001-01-01' and end_change_date='9999-12-31') but still have contacts
    only_one_status_mask = df['person_id'].isin(only_one_status_ids)
    df.loc[only_one_status_mask, 'person_state_code'] = 'A'

    only_one_status_df = df[only_one_status_mask]
    
    ## Patients that changed status
    changed_status_df = df[~only_one_status_mask]
    
    
    mask = changed_status_df['person_state_code'].isin({'I', 'T'}) 
    mask &= (changed_status_df['start_change_date'].apply(get_year) <= end_year) & (changed_status_df['end_change_date'].apply(get_year) >= start_year)
    inactive_ids = set(changed_status_df.loc[mask, 'person_id'].unique())
    n_inactive_1 = len(inactive_ids)
    stats.update({'n_inactive_1':n_inactive_1})
    print(f'\t[INFO] {n_inactive_1:,} patients were inactive at least once during the period of interest (see inactive_ids)')
    
    ## Last status (due to patients reviving, we consider only the last known status)
    changed_status_df = changed_status_df.sort_values(['person_id', 'start_change_date']).groupby(by='person_id').last().reset_index()
    changed_status_df['start_change_date'] = pd.to_datetime(changed_status_df['start_change_date'])
    
    # dead before end_year
    status_before_mask = changed_status_df['start_change_date'] <= end_date
    dead_before_mask = status_before_mask & changed_status_df['person_state_code'].isin({'D', 'S', 'P'})
    dead_ids = changed_status_df.loc[dead_before_mask, 'person_id']
    n_died_be = len(dead_ids)
    stats.update({'n_died_be':n_died_be})
    print(f'\t[INFO] {n_died_be:,} patients died before {end_year+1}')

    if n_died_be > 0:
        print(f'\tExcluding dead patients dead before {end_year+1}...')
        changed_status_df = changed_status_df.loc[~dead_before_mask]

    # inactive / temporaire before end_year
    inactive_before_mask = status_before_mask & changed_status_df['person_state_code'].isin({'I', 'T'})
    inactive_ue_ids = changed_status_df.loc[inactive_before_mask, 'person_id']
    n_inactive_ue = len(inactive_ue_ids)
    stats.update({'n_inactive_ue':n_inactive_ue})
    print(f'\t[INFO] {n_inactive_ue:,} patients who became inactive or temporaire  before {end_year+1} and remained in that state')
    
    if n_inactive_ue > 0:
        print(f'\tExcluding inactive or temporaire patients before {end_year+1} who remained in that state...')
        changed_status_df = changed_status_df.loc[~inactive_before_mask]
    
    ## time to event
    changed_status_df['duration (days)'] = (changed_status_df['start_change_date']-end_date).dt.days
    
    cols = ['person_id', 'person_state_code', 'duration (days)']
    final_df = pd.concat([only_one_status_df, changed_status_df])[cols]
    
    # correct the time to event for active people
    # from NaN if only this status or int if became I/T/D during their life to active_days
    active_days = (extraction_date - end_date).days
    mask = final_df['person_state_code'] == 'A'
    final_df.loc[mask, 'duration (days)'] = active_days
    
    return final_df, stats, inactive_ids