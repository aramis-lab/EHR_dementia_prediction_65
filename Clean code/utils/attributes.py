"""
Functions to retrieve BMI, CCI attributes and also alcohol and tobacco consumption
"""

import pandas as pd
from typing import Tuple, Dict
from .medications import get_year

DataFrameStatsTuple = Tuple[pd.DataFrame, Dict[str, int]]

## Utils
def filter_by_attribute_name(df, col, attr):
    filtered_df = df[df[col].apply(lambda x: x.startswith(attr))]
    filtered_df.drop(columns=[col], inplace=True)
    return filtered_df

def convert_date_to_year(df):
    df['start_change_year'] = df.pop('start_change_date').apply(get_year)
    df['end_change_year'] = df.pop('end_change_date').apply(get_year)
    return df

def compute_mean_value(df, start_year, end_year, name):
    """
    Compute the mean value in the 'value' column of the df, for each person, over a time window. 

    Args:
        - df (pd.DataFrame): A DataFrame containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `value` (float): numeric value associated with the observation.
            * `start_change_year` (int): integer representing the start year of the observation.
            * `end_change_year` (int): integer representing the end year of the observation.
        - start_year (int): An integer representing the start of the time window.
        - end_year (int): An integer representing the end of the time window.
        - name (str): A string representing the name of the variable to compute the mean of.

    Returns:
        - pd.DataFrame: A DataFrame containing two columns: 'person_id' and 'avg. {name}', where the 
        latter is the average value of the specified variable, for each person, over the time window. 
    """
    def helper(data):
        # Get mean over time window (start_year, end_year)
        mask = (data['start_change_year'] > end_year )| (data['end_change_year'] < start_year)
        filtered_rows = data[~mask]
        if not filtered_rows.empty:
            return filtered_rows['value'].mean()

        # Get value at end_year+1 if it exists
        next_year_data = data[(data['start_change_year'] == end_year+1)]
        if not next_year_data.empty:
            return next_year_data.iloc[0]['value']

        # Get value before start_year if it exists
        prev_years_data = data[(data['end_change_year'] < start_year)]
        if not prev_years_data.empty:
            return prev_years_data.iloc[-1]['value']

        # o/w return None
        return None

    groups = df.sort_values(by=['person_id', 'start_change_year']).groupby(by='person_id')
    result = groups.apply(helper).reset_index(name='avg. '+name)
    return result

## Main functions
def get_alcohol_tobacco_consumption(df, start_year, end_year):
    """
    Returns a dataframe containing the average alcohol and tobacco consumption the time window
    defined by start_year and end_year.

    Args:
        - df (pd.DataFrame): The input dataframe containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `person_measure_label` (str): String indicating the type type of measure being taken for a patient. 
              For example, it could be 'Alcohol (glasses/day)', 'Body Mass Index (BMI)', 'Height (cm)', etc.
            * `value` (float): value corresponding to the measure in `person_measure_label`
            * `start_change_date` (str): start date of the observation.
            * `end_change_date` (str): end date of the observation.
        - start_year (int): An integer representing the start year of the time window for which the 
        mean consumption values should be computed.
        - end_year (int): An integer representing the end year of the time window for which the mean 
        consumption values should be computed.

    Returns:
        - A dataframe containing the mean daily alcohol consumption (in glasses/day) and mean daily tobacco 
        consumption (in cigarettes/day) within the time window defined by start_year and end_year.
        - A dictionary containing additional statistics:
            * `n_valid_alcohol` (int): The number of unique patients having at least one valid alcohol record
            * `n_valid_tobacco` (int): The number of unique patients having at least one valid tobacco record
    """
    # Date formatting
    df = convert_date_to_year(df)
    
    # Extract data
    alcohol = filter_by_attribute_name(df, 'person_measure_label', 'Alcohol')
    alcohol = compute_mean_value(alcohol, start_year=start_year, end_year=end_year, name='Alcohol (glasses/day)')
    n_valid_alcohol = alcohol[alcohol.iloc[:, 1].notna()]['person_id'].nunique()
    print(f"\t[INFO] {n_valid_alcohol:,} patients with at least one valid alcohol record")
    
    tobacco = filter_by_attribute_name(df, 'person_measure_label', 'Tobaco')
    tobacco = compute_mean_value(tobacco, start_year=start_year, end_year=end_year, name='Tobaco (cigarettes/day)')
    n_valid_tobacco = tobacco[tobacco.iloc[:, 1].notna()]['person_id'].nunique()
    print(f"\t[INFO] {n_valid_tobacco:,} patients with at least one valid tobacco record")
    
    stats = {'n_valid_alcohol':n_valid_alcohol, 'n_valid_tobacco':n_valid_tobacco}
    return alcohol.merge(tobacco, how='outer'), stats

def get_median_height(df):
    height = filter_by_attribute_name(df, col='person_measure_label', attr='Height')
    height.drop(columns=['person_age', 'start_change_year', 'end_change_year'], inplace=True)
    
    median_height = height.groupby(by='person_id')['value'].median()
    median_height = pd.DataFrame(median_height)
    median_height.rename(columns={'value':'median Height (cm)'}, inplace=True)
    median_height.reset_index(inplace=True)
    return median_height

def get_bmi(df, start_year:int, end_year:int, valid_height_range:Tuple[int, int], 
            valid_weight_range:Tuple[float, float])->DataFrameStatsTuple:
    """
    This function calculates the average Body Mass Index (BMI) of patients within a given time window.
    Due to observed absurd BMI values, the function recalculates the BMI from scratch using the median 
    height taken from all available records for robustness. Patients without a record for height are 
    filtered out, and patients with a height or weight outside of specified ranges are also filtered out, 
    if applicable. The resulting BMI values are then averaged over the time window for each patient.
    
    Args:
        - df: A pandas DataFrame containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `person_measure_label` (str): String indicating the type type of measure being taken for a patient. 
              For example, it could be 'Alcohol (glasses/day)', 'Body Mass Index (BMI)', 'Height (cm)', etc.
            * `value` (float): value corresponding to the measure in `person_measure_label`
            * `start_change_date` (str): start date of the observation.
            * `end_change_date` (str): end date of the observation.
        - start_year (int): An integer representing the start year of the time window.
        - end_year (int): An integer representing the end year of the time window.
        - valid_height_range Tuple[int, int]: A tuple of two integers representing the minimum and 
          maximum height in cm for the patients to be included.
        - valid_weight_range Tuple[float, float]: A tuple of two floats representing the minimum and 
          maximum weight in kg for the patients to be included.
    
    Returns:
        - A pandas DataFrame containing the average BMI of each patient in the time window.
        - A dictionary containing additional statistics:
            * `n_invalid_BMI` (int): The number of unique patients who has an invalid BMI either due to invalid 
            height or weight.
    """
    ### Filtering useless rows and date formatting
    df = df[ df['person_measure_label'].isin(['Height (cm)', 'Weight (kg)']) ]
    df = convert_date_to_year(df)
    
    ### Recalculate BMI
    ## Height
    median_height = get_median_height(df)
    mask = median_height['median Height (cm)'].isna()
    n = mask.sum()
    print(f'\t[INFO] {n:,} patients without a record for height')
    # filtering invalid height
    if n > 0: 
        print('\tFiltering them out...')
        median_height = median_height[~mask]
    if valid_height_range:
        min_height, max_height = valid_height_range
        mask = (median_height['median Height (cm)'] < min_height) | (median_height['median Height (cm)'] > max_height)
        n_invalid_height = sum(mask)
        print(f'\t[INFO] {n_invalid_height:,} patients ({n_invalid_height*100/len(median_height):.2f}%) with a height outside [{min_height} cm, {max_height} cm]')
        if n_invalid_height > 0:
            print('\tFiltering them out...')
            median_height = median_height[~mask]
    
    ## BMI
    weight = filter_by_attribute_name(df, col='person_measure_label', attr='Weight')
    bmi = pd.merge(weight, median_height, on='person_id', how='inner')
    
    if valid_weight_range:
        min_weight, max_weight = valid_weight_range
        mask = (bmi['value'] < min_weight) | (bmi['value'] > max_weight)
        bmi = bmi[~mask]
    
    bmi['value'] = bmi.pop('value') / (bmi.pop('median Height (cm)') / 100) ** 2
    
    bmi = compute_mean_value(bmi, start_year=start_year, end_year=end_year, name='BMI')
    n_invalid_weight = bmi['avg. BMI'].isna().sum()
    print(f'\t[INFO] {n_invalid_weight:,} invalid BMI value ({n_invalid_weight/len(bmi)*100:.2f}%) due to invalid weight')
    
    return bmi, {'n_invalid_BMI':n+n_invalid_height+n_invalid_weight}

def get_charlson(df, start_year, end_year):
    """
    Calculates the average Charlson Comorbidity Index (CCI) for each patient for the given time window.
    
    Args:
        - df (pd.DataFrame): The DataFrame containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `attribute_code` (str): A code representing the measure being taken
            * `value` (float): The value for the given attribute_code
            * `start_change_date` (str): Start date of the observation.
            * `end_change_date` (str): End date of the observation.
        - start_year (int): The starting year of the time window.
        - end_year (int): The ending year of the time window.
    
    Returns:
        - pd.DataFrame: A DataFrame containing two columns: `person_id` and `avg. CHARLSON`, where the 
        latter is the average CCI for each patient over the time window. 
        - A dictionary containing additional statistics:
            * `n_valid_charl` (int): The number of unique patients who have at least one valid CHARLSON record
    
    Notes:
        The Charlson Comorbidity Index (CCI) is a measure of the burden of comorbid disease
        on a patient, and is used to predict mortality. The CCI is based on the presence
        or absence of 17 comorbidities, each with a weighted score. The score for each
        comorbidity is summed to give a total score, which is then used to categorize patients
        into low, moderate, or high-risk groups. It ranges from 0 to 37.
    """
    df = convert_date_to_year(df)
    charlson = filter_by_attribute_name(df, 'attribute_code', 'CHARLSON')
    charlson = compute_mean_value(charlson, start_year=start_year, end_year=end_year, name='CHARLSON')
    n_valid = charlson[charlson['avg. CHARLSON'].notna()]['person_id'].nunique()
    print(f'\t[INFO] {n_valid:,} patients with at least one valid CHARLSON record')
    return charlson, {'n_valid_charl':n_valid}