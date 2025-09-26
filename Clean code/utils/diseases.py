import os
import pandas as pd
import numpy as np
from .dataset import load_pkl
from typing import Tuple, Dict, Set, List

DataFrameStatsTuple = Tuple[pd.DataFrame, Dict[str, int]]

# NEW var: mapping_dir=MAPPING_DIR
def get_icd10(country, mapping_dir):
    """
    Return a dictionary that maps labels to ICD-10 codes for the given country.

    Args:
        - country (str): The country for which the ICD-10 codes are required. Valid values are 'UK' and 'FR'.

    Returns:
        - Dict[str, str]: A dictionary that maps labels to ICD-10 codes.
    """
    mapping = pd.read_csv(os.path.join(mapping_dir, 'mapping.csv'), sep=';')
    # Ne garder que les 2 premières colonnes utiles
    mapping = mapping.iloc[:, :2]
    mapping.columns = ['code', 'content']
    # Supprimer les lignes avec des NaN
    mapping = mapping.dropna()
    content2code = {content.lower():code for _, (code, content) in mapping.iterrows()}
    content2code.update({
        'frontotemporal dementia':'G31.0',
        'memory loss':'R41',
        'memory problem':'R41',
    })
    return content2code

def save_top_missing_codes(df, mapping_dir, country, top_n=10000):
    """
    Sauvegarde le top N des diagnostic_label qui ont toujours diagnostic_icd10 NaN
    
    Args:
        df: DataFrame avec diagnostic_label et diagnostic_icd10
        mapping_dir: répertoire où sauvegarder le fichier
        country: pays pour nommer le fichier
        top_n: nombre de codes à sauvegarder (défaut: 100)
    """
    # Trouver les lignes qui ont toujours diagnostic_icd10 NaN
    missing_codes = df[df['diagnostic_icd10'].isna()]
    
    if len(missing_codes) > 0:
        # Compter la fréquence de chaque diagnostic_label manquant
        top_missing = missing_codes['diagnostic_label'].value_counts().head(top_n)
        
        # Créer un DataFrame pour sauvegarder
        top_missing_df = pd.DataFrame({
            'diagnostic_label': top_missing.index,
            'count': top_missing.values
        })
        
        # Sauvegarder dans un fichier CSV
        output_file = os.path.join(mapping_dir, f'top_{top_n}_missing_icd10_{country}.csv')
        top_missing_df.to_csv(output_file, sep=';', index=False)
        print(f'\t\t[INFO] Top {top_n} missing codes saved to {output_file}')

def complete_icd10(df, country, mapping_dir):
    """
    Completes missing ICD-10 codes in a Pandas DataFrame using a mapping of diagnostic labels to ICD-10 codes
    for the specified country.

    Args:
        - df (pd.DataFrame): A Pandas DataFrame containing the following columns:
            * `person_id` (int): Unique identifier for each person in the dataset.
            * `diagnostic_label` (str): Diagnostic in the form of a sentence
            * `diagnostic_icd10` (str): Code associated to the diagnostic
        - country (str): The country for which the ICD-10 codes are required. Valid values are 'UK' and 'FR'.

    Returns:
        - pd.DataFrame: The input DataFrame with missing ICD-10 codes completed. If any missing values remain, rows
        containing those missing values are removed from the DataFrame.
        - A dictionary containing additional statistics:
            * `icd10_na_rate` (float): The proportion of missing ICD10 codes in the input dataframe.
    """
    print('\tCompleting missing icd10...')
    content2code = get_icd10(country, mapping_dir)
    n_missing = df['diagnostic_icd10'].isna().sum()
    print(f'\t\t[INFO] Initially: {n_missing:,} missing icd10 (~ {n_missing/len(df)*100:.2f}%)')
    new_labels = df['diagnostic_label'].apply(lambda label: content2code.get(label.lower(), np.float64('nan')))
    df['diagnostic_icd10'] = df['diagnostic_icd10'].mask(df['diagnostic_icd10'].isna(),
                                                         new_labels)
    
    # Sauvegarder le top 100 des codes manquants
    save_top_missing_codes(df, mapping_dir, country)
    n_missing=df['diagnostic_icd10'].isna().sum()
    icd10_na_rate = n_missing/len(df)
    print(f'\t\t[INFO] After: {n_missing:,} missing icd10 (~ {icd10_na_rate*100:.2f}%)')
    if n_missing > 0:
        print('\tRemoving the NaNs icd10...')
        df = df[df['diagnostic_icd10'].notna()]
    print({'icd10_na_rate':icd10_na_rate})
    return df, {'icd10_na_rate':icd10_na_rate}


## 
# NEW var: ref_codes=CODES => MODIFICATIONS IN ALL FOLLOWING FUNCTIONS + MAIN FUNCTION
def get_disease_from_code(code, ref_codes):
    """
    Returns a tuple of disease names corresponding to the ICD-10 code provided.

    Args:
        - code (str): A string representing an ICD-10 code.

    Returns:
        - Tuple[str]: A tuple containing disease names corresponding to the ICD-10 code provided. Valid disease
        names can be found in CODES.keys() (examples: 'parkinson', 'alzheimer', 'vascular_dementias', ...''). 
        If the provided code does not correspond to any of these diseases, an empty tuple is returned.
    """
    # ICD-10 coding system format follows the convention of a letter followed by two digits, which may optionally be 
    # followed by a decimal point and one or more digits.
    if len(code) > 3 and code[3] != '.':
        code = code[:3]+'.'+code[3:]
    
    helper = lambda codes: any([code.startswith(c) for c in codes])
    
    res = tuple()
    for disease, icd10_codes in ref_codes.items():
        if helper(icd10_codes): res+=(disease,)
    print(res)
    return res

def get_healthy_people(df, empty_convention)->pd.DataFrame:
    """
    Returns a Pandas DataFrame containing the IDs of individuals who do not have any recorded diseases in the input
    DataFrame.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing at least columns 'person_id' and 'disease'.
        empty_convention: Object that is used to represent an absence of diseases.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing columns 'person_id' and 'diseases'. The 'person_id' column
        contains the IDs of individuals who do not have any recorded diseases, and the 'diseases' column contains
        the empty_convention string for each individual.
    """
    healthy_people = set()
    grouped_df = df.groupby(by='person_id')
    for person_id, group in grouped_df:
        if (group['disease'] == tuple()).all():
            healthy_people.add(person_id)

    healthy_people = pd.DataFrame({
        'person_id':list(healthy_people), 
        'diseases':[empty_convention]*len(healthy_people)
    })
    return healthy_people

def get_diseases(df, country, end_year, mapping_dir, ref_codes=None):
    """
    Returns a dataframe containing the first time each disease was declared for each person, and a dictionary with the
    list of diseases that should be censored for each person.
    
    e.g. If person with id 00000 has parkinson, alzheimer and dementias and only dementias was observed prior 
    to end_year the output will be
    +-----------+----------------------------------------------------------+
    | person_id |                      diseases                            |    {00000:['dementias']}
    +-----------+----------------------------------------------------------+
    |   00000   | [('parkinson', t1_declared), ('alzheimer', t2_declared)] |
    +-----------+----------------------------------------------------------+
    
    Args:
        - df (pd.DataFrame): A pandas DataFrame containing the data to be processed. The DataFrame must have the following
        columns: `person_id` (str), `contact_date` (str representing a date in yyyy-mm-dd format), and `diagnostic_icd10` (str).
        - country (str): A string representing the country code for which to process the data. The valid values are 'FR' and 'UK'.
        - end_year (int): An integer representing the last year of the data to be processed.

    Returns:
        - Tuple[pd.DataFrame, Dict[str, float], Dict[str, List[str]]]: A tuple containing two elements:
            - A pandas DataFrame containing the first time each disease was declared for each person. The DataFrame has two
            columns: 'person_id' (string) and 'diseases' (list of tuples). The 'diseases' column contains a list of tuples
            representing the diseases declared by the person, where each tuple has two elements: the name of the disease
            (string) and the number of days since the end year (int) when the disease was declared.
            - A dictionary containing additional statistics:
                * `n_patients_dis` (int): The number of unique patients in the input dataframe.
                * `icd10_na_rate` (float): The proportion of missing ICD10 codes in the input dataframe.
                * `n_healthy` (int): The number of unique patients having no neurodegenerative  disease the input dataframe.
                * `n_neurodis_be` (int): The number of unique patients having at least one neurodegenerative  disease before 
                start_year in the input dataframe.
            - A dictionary mapping patient_id to the list of names of the diseases that was diagnosed before end_year
    """
    ### Convert icd10 to disease name
    n_patients_dis = df['person_id'].nunique()
    stats = {'n_patients_dis':n_patients_dis}
    print(f'\t[INFO] {n_patients_dis:,} unique patients')
    
    df, stats_ = complete_icd10(df, country, mapping_dir)
    
    # Définir les codes de référence par défaut si non fournis
    if ref_codes is None:
        ref_codes = {
            'parkinson': {'G20', 'G21', 'G22'},
            'alzheimer': {'F00', 'G30'},
            'vascular_dementias': {'F01'},
            'dementias': {'F03'},
            'mci': {'F06.7', 'R41'}  # Mild Cognitive Impairment
        }
    
    df['disease'] = df.pop('diagnostic_icd10').apply(lambda code: get_disease_from_code(code, ref_codes))
    stats.update(stats_)
    
    print('\tProcessing diseases...')
    ### Split diseases 
    # Since they are a set of names, we will create a new row for each element of the set    
    
    ## Separate healthy and sick people (optimization procedure)
    healthy_people = get_healthy_people(df, empty_convention=[])
    n_healthy = len(healthy_people)
    stats.update({'n_healthy':n_healthy})
    print(f'\t[INFO] {n_healthy:,} patients without neurodegenerative diseases')
    
    mask = df['person_id'].isin(healthy_people['person_id']) | (df['disease'] == tuple())
    df_sick = df.loc[~mask]
    diseases = df_sick['disease'].explode()
    
    ## Split diseases
    df_diseases = pd.DataFrame(columns=df.columns)
    for person_id, disease in zip(diseases.index, diseases.values):
        new_row = df_sick.loc[person_id].copy()
        new_row['disease'] = disease
        df_diseases = pd.concat([df_diseases, new_row.to_frame().transpose()])
    
    ### Extract data from sick people
    # Get time declared
    df_diseases['contact_date'] = pd.to_datetime(df['contact_date'])
    end_date = pd.to_datetime(f'{end_year}-12-31', format='%Y-%m-%d')
    df_diseases['time_declared'] = (df_diseases['contact_date'] - end_date).dt.days
    
    # Find the first time it was declared
    df_diseases = df_diseases[['person_id', 'disease', 'time_declared']]
    grouped_df = df_diseases.groupby(['person_id', 'disease'])

    to_censure = {} # person_id:[disease, ...]
    to_df = {} # person_id:[(disease, first_time_declared), ...]
    mci_at_baseline = set() # person_id qui ont MCI à baseline
    
    for group, data in grouped_df:
        person_id, disease = group[0], group[1]
        
        if disease:
            min_time_date = data['time_declared'].min()
            if min_time_date <= 0:
                # Ne pas censurer les patients avec MCI même s'ils ont été diagnostiqués avant end_year
                if disease not in  ['mci','all_dementias+mci']:
                    to_censure[person_id] = to_censure.get(person_id, []) + [disease]
                else:
                    # Marquer les patients avec MCI à baseline
                    mci_at_baseline.add(person_id)
                    # Traiter MCI comme une maladie à prédire même si diagnostiquée avant end_year
                    to_df[person_id] = to_df.get(person_id, []) + [(disease, min_time_date)]
            else:
                to_df[person_id] = to_df.get(person_id, []) + [(disease, min_time_date)]
    print(f'\t[INFO] {len(to_censure):,} patients had a neurodegenerative disease diagnosed\n\tbefore end_year')
    stats.update({'n_neurodis_be':len(to_censure)})
    
    # Formatting
    new_df = {'person_id':[], 'diseases':[]}
    for person_id, diseases in to_df.items():
        new_df['person_id'].append(person_id)
        new_df['diseases'].append(diseases)
    new_df = pd.DataFrame(new_df)
    
    ## Add healty people to `new_df`
    new_df = pd.concat([new_df, healthy_people], ignore_index=True)
    
    # Ajouter la colonne MCI à baseline pour tous les patients (0/1 au lieu de True/False)
    new_df['mci_at_baseline'] = new_df['person_id'].apply(lambda pid: 1 if pid in mci_at_baseline else 0)
    
    print(f'\t[INFO] {len(mci_at_baseline):,} patients had MCI at baseline')
    stats.update({'n_mci_baseline': len(mci_at_baseline)})
    
    return new_df, stats, to_censure