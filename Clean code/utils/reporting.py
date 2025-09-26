import numpy as np
from typing import Dict, List

# NEW var: diseases_of_interest=DISEASES_OF_INTEREST
def count_diseases(df, diseases_of_interest)->Dict[str, int]:
    count = {}
    for disease in diseases_of_interest+['MCI + all_dementias']:
        # new
        if disease == 'MCI + all_dementias':
            count[disease] = df['diseases'].apply(lambda x: any(d in {'mci', 'all_dementias'} for d, _ in x)).sum()
        else:
            count[disease] = df['diseases'].apply(lambda x: any(d == disease for d, _ in x)).sum()
    return count


rename = lambda name: ' '.join(name.split('_')).capitalize()

# NEW var: diseases_of_interest=DISEASES_OF_INTEREST
def get_general_info(df, to_censure, diseases_of_interest):
    ## Count diseases
    # Patients who had another disease before
    before_person_ids = set(to_censure.keys())
    before_diseases = []
    for person_id in before_person_ids:
        l = to_censure[person_id]
        # new
        if ('mci' in l) or ('all_dementias' in l):
            l += ['MCI + all_dementias']
        
        before_diseases.extend(l)

    # Patients who have a disease after the period of interest
    count:Dict[str, int] = count_diseases(df)
    
    ## DataFrame
    s = set(df['person_id'])
    n = len(s.union(before_person_ids))
    n_ = len(s)
    new_df = pd.DataFrame(columns=[DISEASES_OF_INTEREST])
    for disease in diseases_of_interest+['MCI + all_dementias']:
        c1, c2 = count[disease], before_diseases.count(disease)
        total = c1+c2
        new_df[disease] = [c1, np.round(c1/n_*1000,2), 
                           c2, total, np.round(total/n*1000,2)]
    new_df.index = ['Dataset', 'Dataset Portion (‰)', 'Disease before', 'Total','Total Portion (‰)']
    new_df.rename(columns={disease:rename(disease) for disease in diseases_of_interest}, inplace=True)
    return new_df

def get_gender_count(df, m=None):
    n = len(df)
    n_f = df['gender_code'].eq('F').sum()
    n_m = df['gender_code'].eq('M').sum()
    data = [f'{n:,}', 
            f'{n_f:,} ({get_pct(n_f, n)})', 
            f'{n_m:,} ({get_pct(n_m, n)})']
    if m and m > 0:
        data[0] = f'{n:,} ({n/m*100:.2f}%)'
    return data

def get_time_to_disease(df, disease, to_year=True):
    helper = lambda x: [t for d, t in x if d == disease][0]
    time_to_event = df['diseases'].apply(helper) 
    if len(time_to_event) == 0: return ['']
    
    if to_year:
        time_to_event /= 365.25
    q1, median, q3 = np.percentile(time_to_event, [25, 50, 75])
    return [f'{disease}: {median:.2f} ({q1:.2f} - {q3:.2f})']

def get_time_to_status(df, status, to_year=True):
    time_to_event = df['duration (days)'] 
    if len(time_to_event) == 0: return ['']
    
    if to_year:
        time_to_event /= 365.25
    q1, median, q3 = np.percentile(time_to_event, [25, 50, 75])
    return [f'{status}: {median:.2f} ({q1:.2f} - {q3:.2f})']

def get_most_used_meds(df, m)->List[int]:
    helper = lambda n: f'{n/m*100:.2f}%' if m > 0 else ' - ' 
    return [ f'{n:,} ({helper(n)})' for n in df.iloc[:,9:].sum().astype(int)]

def get_pct(n1, n2):
    if n2 == 0: return ''
    return f'{n1/n2 * 100:.2f}%'

str2int = lambda n: int(n.replace(',', ''))

def summarize(df, disease, to_year=True):
    ### Total population
    tot_pop_gender_count = get_gender_count(df)
    meds_names = df.columns[9:]
    n = str2int(tot_pop_gender_count[0])
    tot_pop_data = tot_pop_gender_count + ['-']*2 + get_most_used_meds(df, n)

    ### Disease-specific
    disease_mask = df['diseases'].apply(lambda x: any(d == disease for d, _ in x))
    disease_df = df[disease_mask]
    
    def helper(df, time_to_disease=False, status_name=None):
        res = get_gender_count(df, n)
        res += get_time_to_disease(df, disease, to_year) if time_to_disease else ['-']
        res += get_time_to_status(df, status_name, to_year) if status_name else ['-']
        res += get_most_used_meds(df, str2int(res[0].split(' (')[0]))
        return res
    
    ## All
    disease_data = helper(disease_df, time_to_disease=True)
    
    ## Disease and alive
    disease_alive_df = disease_df[disease_df['person_state_code'].eq('A')]
    disease_alive_data = helper(disease_alive_df, time_to_disease=True) 
    
    ## Disease and death
    disease_death_df = disease_df[disease_df['person_state_code'].isin({'D', 'P', 'S'})]
    disease_death_data = helper(disease_death_df, time_to_disease=True, status_name='death')
    
    ## Dead w/o disease
    alive_wo_disease_df = df[df['person_state_code'].eq('A') & (~disease_mask)]
    alive_wo_disease_data = helper(alive_wo_disease_df) 
    
    ## Dead w/o disease
    dead_wo_disease_df = df[(df['person_state_code'].isin({'D', 'P', 'S'}) & (~disease_mask))]
    dead_wo_disease_data = helper(dead_wo_disease_df, status_name='death') 
    
    ## Censored before disease or death
    disease_df_ = disease_df[disease_df['person_state_code'].isin({'I', 'T'})]
    mask_sick_before_censored = disease_df_['diseases'].apply(lambda x: [t for d, t in x if d==disease][0]) < disease_df_['duration (days)']
    ids_sick_before_censored = disease_df_.loc[mask_sick_before_censored, 'person_id']
    
    mask = df['person_state_code'].isin({'I', 'T'}) & (~df['person_id'].isin(ids_sick_before_censored))
    censored_before_disease_or_death_df = df[mask]
    censored_before_disease_or_death_data = helper(censored_before_disease_or_death_df, status_name='Inactive or Temporaire')
    
    ### Create summary dataframe
    disease = rename(disease)
    summary = pd.DataFrame({
        'tot. population': tot_pop_data,
        
        # Disease
        disease:disease_data,
        f'{disease} and alive': disease_alive_data,
        f'{disease} and dead': disease_death_data,
        
        # Other
        f'alive w/o {disease}': alive_wo_disease_data,
        f'dead w/o {disease}': dead_wo_disease_data,
        f'Censored before {disease} or death (I, T)': censored_before_disease_or_death_data,
    }, index=['n', 'n_f', 'n_m', 'Time to event', 'event: med (q1 - q3)'] + list(meds_names))
    
    return summary


