#!/usr/bin/env python3
"""
Script de test pour la fonction Kaplan-Meier
"""

import pandas as pd
import numpy as np
from kaplan_meier_function import build_kaplan_meier, compare_kaplan_meier


def create_sample_data(n_patients=1000, seed=42):
    """
    Créer des données d'exemple au format preprocessed_df
    """
    np.random.seed(seed)
    
    # Générer des données simulées
    data = {
        'time': np.random.exponential(scale=1000, size=n_patients),  # Temps en jours
        'event': np.random.choice([0, 1, 2], size=n_patients, p=[0.6, 0.3, 0.1]),  # 0=censuré, 1=maladie, 2=décès
        'is female': np.random.choice([0, 1], size=n_patients, p=[0.45, 0.55]),
        'avg. CHARLSON': np.random.normal(2, 1, n_patients).clip(0, None),
        'avg. BMI': np.random.normal(25, 4, n_patients).clip(15, 45),
        'drink ≥ 2 glasses/day': np.random.choice([0, 1], size=n_patients, p=[0.8, 0.2]),
        'smoke': np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3]),
    }
    
    # Ajouter quelques médicaments simulés
    for med in ['A06', 'C09', 'C10', 'N06']:
        data[med] = np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3])
    
    df = pd.DataFrame(data)
    
    # S'assurer que les temps sont positifs
    df['time'] = df['time'].clip(1, None)
    
    return df


def test_kaplan_meier_function():
    """
    Tester la fonction build_kaplan_meier
    """
    print("=== Test de la fonction build_kaplan_meier ===\n")
    
    # Créer des données d'exemple
    preprocessed_df = create_sample_data()
    
    print(f"Données créées: {len(preprocessed_df)} patients")
    print(f"Distribution des événements:")
    print(f"  - Censurés (0): {(preprocessed_df['event'] == 0).sum()}")
    print(f"  - Maladie (1): {(preprocessed_df['event'] == 1).sum()}")
    print(f"  - Décès (2): {(preprocessed_df['event'] == 2).sum()}")
    print()
    
    # Test 1: Kaplan-Meier simple
    print("1. Test Kaplan-Meier simple pour 'alzheimer'")
    kmf = build_kaplan_meier(preprocessed_df, 'alzheimer', show_plot=False)
    print(f"   Temps médian: {kmf.median_survival_time_}")
    print()
    
    # Test 2: Comparaison par sexe
    print("2. Test de comparaison par sexe")
    kmf_groups = compare_kaplan_meier(
        preprocessed_df, 
        'alzheimer', 
        'is female',
        group_labels={0: 'Hommes', 1: 'Femmes'},
        show_plot=False
    )
    print(f"   Groupes créés: {list(kmf_groups.keys())}")
    print()
    
    # Test 3: Comparaison par consommation d'alcool
    print("3. Test de comparaison par consommation d'alcool")
    kmf_alcohol = compare_kaplan_meier(
        preprocessed_df, 
        'alzheimer', 
        'drink ≥ 2 glasses/day',
        group_labels={0: 'Buveurs modérés', 1: 'Gros buveurs'},
        show_plot=False
    )
    print(f"   Groupes créés: {list(kmf_alcohol.keys())}")
    print()
    
    print("✅ Tous les tests sont passés avec succès!")


def example_usage():
    """
    Exemple d'utilisation avec vos vraies données
    """
    print("\n" + "="*60)
    print("EXEMPLE D'UTILISATION AVEC VOS DONNÉES")
    print("="*60)
    
    example_code = '''
# Exemple d'utilisation avec vos vraies données preprocessed_df

from kaplan_meier_function import build_kaplan_meier, compare_kaplan_meier

# 1. Courbe de Kaplan-Meier simple pour Alzheimer
kmf = build_kaplan_meier(preprocessed_df['alzheimer']['UK_70'], 'alzheimer')

# 2. Comparaison hommes vs femmes pour Alzheimer
kmf_gender = compare_kaplan_meier(
    preprocessed_df['alzheimer']['UK_70'], 
    'alzheimer', 
    'is female',
    group_labels={0: 'Hommes', 1: 'Femmes'}
)

# 3. Comparaison avec un médicament (par exemple A06)
kmf_med = compare_kaplan_meier(
    preprocessed_df['alzheimer']['UK_70'], 
    'alzheimer', 
    'A06',
    group_labels={0: 'Sans A06', 1: 'Avec A06'}
)

# 4. Obtenir des statistiques sans afficher le graphique
kmf_stats = build_kaplan_meier(
    preprocessed_df['parkinson']['FR_65'], 
    'parkinson', 
    show_plot=False
)
print(f"Médiane de survie: {kmf_stats.median_survival_time_} jours")

# 5. Personnaliser le graphique
kmf_custom = build_kaplan_meier(
    preprocessed_df['mci']['UK_65'], 
    'MCI',
    title="Courbe de Kaplan-Meier personnalisée - MCI",
    xlim=(0, 3650),  # Limiter à 10 ans
    alpha=0.01  # Intervalles de confiance à 99%
)
'''
    
    print(example_code)


if __name__ == "__main__":
    test_kaplan_meier_function()
    example_usage()