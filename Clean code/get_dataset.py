import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set

from utils.dataset import load_thin_datasets
# Il faut définir le répertoire de base des données
import os
base_dir = os.path.join('..', '..', '..', 'data', 'Extractions_EU')
from utils.sanity_check import check_valid_input
from utils.gender import remove_undefined_gender
from utils.medications import get_one_hot_encoded_medications
from utils.attributes import get_charlson, get_bmi, get_alcohol_tobacco_consumption
from utils.status import get_status
from utils.diseases import get_diseases
from utils.medications import get_year

def get_dataset(country:str, age:int,
                n_most_frequent:int=None,
                min_before:int=0, min_during:int=1, min_after:int=1,
                valid_height_range=None, valid_weight_range=None,
                start_year:int=2008, end_year:int=2010, 
                extraction_date:str='2023-01-09', mapping_dir:str=None, ref_codes=None):
    """
    Generate a dataset of patients from the given country and age, with the following features:
    - gender (male or female)
    - one hot encoding of medications taken between the start_year and end_year
    - BMI, CHARLSON, Alcohol, Tobacco (average between start_year and end_year if available, 
      otherwise we take the closest value i.e. value of end_year+1 or first before start_year)
    
    The target is extracted from the CONTACT_DIAGNOSTICS table where the practitioner has detected at 
    least once the disease. The output is in the format [(disease_name, time), ..., (disease_name, time)].
    %TODO: more robust at least twice
    
    Args:
        - country (str): The country of interest. Case insensitive. Currently supports UK and FR.
        - age (int): The age of patients of interest. Either 65 or 70.
        - n_most_frequent (int | None): The number of most frequent medications to consider for one 
          hot encoding. Default is None (keeping all columns).
        - min_before (int): The minimum count of contacts required before end_year to be considered valid patient. Default is 0.
        - min_during (int): The minimum count of contacts required during the time window to be considered valid patient. Default is 1.
        - min_after (int): The minimum count of contacts required after the end_year to be considered valid patient. Default is 1.
        - valid_height_range (Tuple[int, int] | None):  A tuple representing the range of acceptable heights (in cm) 
        for patients to calculate BMI. Default is (100, 250) corresponding to limitation set by the following paper:
        'Assessment of Risk Factors and Early Presentations of Parkinson Disease in Primary Care in a Diverse UK Population'. 
        - valid_weight_range (Tuple[float, float] | None): A tuple representing the range of acceptable weights (in kg) 
        for patients to calculate BMI. Default is (30, 250) corresponding to limitation set by the following paper:
        'Assessment of Risk Factors and Early Presentations of Parkinson Disease in Primary Care in a Diverse UK Population'. 
        - start_year (int): The start year to consider for feature extraction. Default is 2008.
        - end_year (int): The end year to consider for feature extraction. Default is 2010.  
        - extraction_date (str): The date of data extraction, in the format 'yyy-mm-dd'. Default is '2023-01-09'.
    
    Returns:
        - A pandas DataFrame with the hot encoded columns, indicating whether the person has taken that medication
        during the considered time window, for the most frequent medications (first 3 letters) as well as the 
        following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `diseases` (List[Tuple[str, int]]): representing the diseases and time declared by the practitioner.
            Each tuple contains two elements: a string representing the name of the disease, and an integer 
            representing the number of days since the specified end_year when the disease was declared. 
            The diseases of interest  are 'parkinson', 'alzheimer', 'vascular_dementias', and 'dementias'. If a 
            disease of interest is observed in the patient before `end_year` we will censor this person (see 
            get_disease function for more details).
            * `gender_code` (str): gender either 'F' (female) or 'M' (male).
            * `person_state_code` (str): final state the person is in. The possible values are:
            'A' (Active), 'I' (Inactive), 'D' (Dead), 'T' (Temporaire, specific to UK), 'P' (Probably dead, specific
              to France) and 'S' (Suspected dead, specific to France)
                %TODO: explain the states 
            * `duration (days)` (int): number of days after the specified end_year that the person reached the 
              final state given by the column person_state_label
            * `avg. Alcohol (glasses/day)` (float): mean Alcohol over a time window defined by start_year and end_year
            * `avg. Tobaco (cigarettes/day)` (float): mean Tobaco over a time window defined by start_year and end_year
            * `avg. BMI` (float): mean BMI over a time window defined by start_year and end_year
            * `avg. CHARLSON` (float): mean CHARLSON over a time window defined by start_year and end_year
        - has_another_disease_before (Dict[int, List[str]]): A dictionary mapping person_id to the list of diseases that 
        was detected before end_year.
        - inactive_ids (Set[int]): A set of ids of patient being inactive at least once during the period of interest.
            
    Notes:
        - AssertionError: If any of the input parameters are invalid.
        - For more details see the functions
    """

    country = country.upper()
    check_valid_input(country, age, valid_height_range, valid_weight_range, extraction_date, min_before, min_during, min_after)
    
    #==============================================================================================================
    # Load datasets
    #==============================================================================================================
    names = [
        # filter out inactive people
        ('CONTACT', ['person_id', 'contact_date', 'contact_type_code']),
        
        # diseases
        ('CONTACT_DIAGNOSTICS', ['person_id', 'contact_id', 'contact_date', 'diagnostic_label', 'diagnostic_icd10']),
        # genre
        ('PERSON', ['person_id', 'gender_code']), 
        # medications  -> [:3]
        ('CONTACT_PRESCRIPTIONS', ['person_id', 'contact_date',  'product_atc_code']), 
        # status (censoring)
        ('SOCIAL_CHANGES', ['person_id', 'end_change_date', 'start_change_date', 'person_state_code']), 
        # Alcohol, Tobacco
        ('MEASURE_CHANGES', ['person_id', 'person_measure_label', 'value', 'start_change_date', 'end_change_date']), 
        # CHARLSON
        ('ATTRIBUTES_CHANGES', ['person_id', 'attribute_code', 'value', 'start_change_date', 'end_change_date']),
        # BMI
        ('PERSON_MEASURE_CHANGES', ['person_id', 'person_age', 'person_measure_label', 'value', 
                                    'start_change_date', 'end_change_date']), 
    ]
        
    print('* Begin datasets loadings:\n')    
    dfs = load_thin_datasets(
        names=names, country=country, age=age, base_dir=base_dir
    )
    
    # renaming columns / attributes in the df
    if country == 'FR':
        # Alcohol, Tobacco
        df = dfs['MEASURE_CHANGES']
        df['person_measure_label'] = df['person_measure_label'].replace({'Tabac (cigarette/jour)':'Tobaco (cigarettes/day)', 
                                                                         'Alcool (verre/jour)':'Alcohol (glasses/day)'})
        # BMI
        df = dfs['PERSON_MEASURE_CHANGES']
        df['person_measure_label'] = df['person_measure_label'].replace({'Taille (cm)':'Height (cm)', 
                                                                         'Poids (kg)':'Weight (kg)',})
    
    ## Filtering out inactive people from all datasets
    contact_df = dfs.pop('CONTACT')
    
    # keep only contacts not corresponding to lab results
    contact_df = contact_df.loc[~contact_df['contact_type_code'].eq('R')]
    contact_df['contact_year'] = contact_df.pop('contact_date').apply(get_year)
    
    # get active ids
    def get_valid_window_ids(df, min_count, before=None, after=None):
        """
        Returns IDs of patients who have at least `min_count` contacts during the time window [before, after]
            - (-inf, after] if after is set to None
            - [before, +inf) if after is set to None
            - [before, after] if both are integers
        """
        if min_count == 0: return  df['person_id'].unique()
        
        time_mask = (df['contact_year'] <= before) if before else np.ones(len(df), dtype=bool)
        time_mask &= (df['contact_year'] >= after) if after else np.ones(len(df), dtype=bool)
        return df.loc[time_mask, 'person_id'].value_counts().loc[lambda x: x >= min_count].index
    
    before_ids = get_valid_window_ids(contact_df, min_count=min_before, before=end_year)
    during_ids = get_valid_window_ids(contact_df, min_count=min_during, before=end_year, after=start_year)
    after_ids = get_valid_window_ids(contact_df, min_count=min_after, after=end_year+1)

    active_ids = set(before_ids) & set(during_ids) & set(after_ids)
    
    # filtering all the other datasets
    n = dfs['PERSON']['person_id'].nunique()
    for key, df in dfs.items():
        df = df[df['person_id'].isin(active_ids)]
        dfs[key] = df
    
    n_new =  dfs['PERSON']['person_id'].nunique()
    print(f'* Filtering out {n-n_new:,} inactive patients ({100*(1-n_new/n):.2f}% of {n})\n')
    stats = {'insufficient_contacts':n-n_new}
    
    print('* Begin extraction of interesting features:\n'
          'Note: statistics display using [INFO] flag is from the current table only and are not combined ' 
          'with info from other tables\n')
    #==============================================================================================================
    # Feature extractions
    #==============================================================================================================
    print('  > GENDER:')
    gender, stats_ = remove_undefined_gender(dfs['PERSON'])
    stats.update(stats_)
    
    print('\n\n  > STATUS:') 
    status, stats_, inactive_ids = get_status(df=dfs['SOCIAL_CHANGES'], 
                                      start_year=start_year, end_year=end_year,
                                      extraction_date=extraction_date)
    X = gender.merge(status, on='person_id', how='left')
    stats.update(stats_)
    # alcohol, tabacco
    print('\n\n  > Alcohol and tabacco consumption:') 
    alcohol_tobacco, stats_ = get_alcohol_tobacco_consumption(df=dfs['MEASURE_CHANGES'],
                                                      start_year=start_year, end_year=end_year,)
    X = X.merge(alcohol_tobacco, on='person_id', how='left')
    stats.update(stats_)
    
    # BMI
    print('\n\n  > BMI:')
    bmi, stats_ = get_bmi(df=dfs['PERSON_MEASURE_CHANGES'],
                  start_year=start_year, end_year=end_year,
                  valid_height_range=valid_height_range, valid_weight_range=valid_weight_range,)
    X = X.merge(bmi, on='person_id', how='left')
    stats.update(stats_)
    
    # CHARLSON
    print('\n\n  > CHARLSON:')
    charlson, stats_ = get_charlson(df=dfs['ATTRIBUTES_CHANGES'],
                            start_year=start_year, end_year=end_year)
    X = X.merge(charlson, on='person_id', how='left')
    stats.update(stats_)

    print('\n\n  > MEDICATIONS:')
    medications, stats_ = get_one_hot_encoded_medications(dfs['CONTACT_PRESCRIPTIONS'],
                                                  start_year=start_year, end_year=end_year, 
                                                  n_most_frequent=n_most_frequent)
    X = X.merge(medications, on='person_id', how='left')
    stats.update(stats_)
    
    #==============================================================================================================
    # Target extraction
    #==============================================================================================================
    print('\n\n  > DISEASES:')
    diseases, stats_, has_another_disease_before = get_diseases(df=dfs['CONTACT_DIAGNOSTICS'], 
                                                        country=country, end_year=end_year, mapping_dir=mapping_dir,ref_codes=ref_codes)
    stats.update(stats_)
    df = diseases.merge(X, on='person_id', how='right')
    return df, has_another_disease_before, inactive_ids, stats

