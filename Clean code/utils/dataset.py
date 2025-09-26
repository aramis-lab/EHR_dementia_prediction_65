import os
import pandas as pd
import pickle
from glob import glob
from typing import List, Tuple, Dict

# THIN database
def read_csv(file_name, base_dir, sep=';', usecols=None,low_memory=False, print_load=None)->pd.DataFrame:
    if print_load: print(f'  > Loading: {print_load}...')
    df = pd.read_csv(os.path.join(base_dir, file_name), sep=sep, usecols=usecols, low_memory=low_memory)
    if print_load: 
        n_unique = df['person_id'].nunique()
        print(f'\t[INFO] {len(df):,} samples for {n_unique:,} unique patients\n')
    return df

def get_file_name(name:str, country:str, age:int, base_dir)->str:
    """ Find the first file that match: `base_dir/country_age/*_name.csv` """
    return glob(os.path.join(base_dir, f'{country}_{age}', f'*_{name}.csv'))[0]

def load_thin_datasets(names:List[Tuple[str, List[str]]], country:str, age:int, base_dir:str)->Dict[str, pd.DataFrame]:
    """
    Load multiple datasets into memory as pandas dataframes.

    Args:
        - names (List[Tuple[str, List[str]]]): A list of tuples, where each tuple represents a dataset to load.
          The first element of the tuple is the name of the dataset, and the second element is a list of column
          names to keep (the order of `usecols` is not important as it is rearranged automatically when loading
          the CSV files).
        - country (str): The name of the country for which to load the datasets.
        - age (int): The age range for which to load the datasets.

    Returns:
        - dfs (Dict[str, pd.DataFrame]): A dictionary mapping dataset names to pandas dataframes.
    """
    dfs = {
        name:read_csv(file_name=get_file_name(name, country=country, age=age, base_dir=base_dir),
                      base_dir='', print_load=name.replace('_', ' '), usecols=usecols) for name, usecols in names
    }
    return dfs

# ICD-10 mapping
def load_pkl(file_name, base_dir)->Dict[str, str]:
    file_name = os.path.join(base_dir, file_name)
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
# Datasets made
def load_dataset(country, age, base_dir, only_df=False):
    dataset_file = os.path.join(base_dir, country + "_" + str(age), "dataset.csv")
    dataset = pd.read_csv(dataset_file)
    # convert the tuple strings back to tuples (when we load the df the `diseases` column is considered as a string)
    dataset['diseases'] = dataset['diseases'].apply(lambda x: ast.literal_eval(x))
    
    if only_df:
        return dataset
    else:
        cohort = f"{country}_{age}"
        
        to_censure_file = os.path.join(base_dir, cohort, "to_censure.pkl")
        inactive_ids_file = os.path.join(base_dir, cohort, "inactive_ids.pkl")
        stats_file = os.path.join(base_dir, cohort, "stats.pkl")
        
        with open(to_censure_file, "rb") as f:
            to_censure = pickle.load(f)
        with open(inactive_ids_file, "rb") as f:
            inactive_ids = pickle.load(f)
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)
        return dataset, to_censure, inactive_ids, stats