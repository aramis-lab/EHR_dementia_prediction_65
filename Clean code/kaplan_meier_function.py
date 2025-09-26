import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from typing import Optional, Tuple


def build_kaplan_meier(preprocessed_df, disease: str, 
                       country: str, age: int,
                       alpha: float = 0.05, 
                       title: Optional[str] = None,
                       xlim: Optional[Tuple[float, float]] = None,
                       show_plot: bool = True,
                       time_unit: str = 'years') -> KaplanMeierFitter:
    """
    Construire une courbe de Kaplan-Meier pour l'apparition d'une maladie.
    
    Parameters:
    -----------
    preprocessed_df : dict ou pd.DataFrame
        Si dict: dictionnaire structuré preprocessed_df[disease][country_age] contenant les DataFrames
        Si pd.DataFrame: DataFrame préprocessé contenant les colonnes 'time', 'event' et les covariables
        Format attendu:
        - 'time': temps jusqu'à l'événement (en jours)
        - 'event': indicateur d'événement (0=censuré, 1=maladie, 2=décès)
    
    disease : str
        Nom de la maladie d'intérêt
        
    country : str
        Pays ('FR' ou 'UK')
        
    age : int
        Âge de la cohorte (65 ou 70)
        
    alpha : float, default=0.05
        Niveau de confiance pour les intervalles de confiance (1-alpha)
        
    title : str, optional
        Titre personnalisé pour le graphique
        
    xlim : tuple, optional
        Limites de l'axe x (min_time, max_time) dans l'unité spécifiée
        
    show_plot : bool, default=True
        Afficher le graphique ou non
        
    time_unit : str, default='years'
        Unité de temps pour l'affichage ('years' ou 'days')
        
    Returns:
    --------
    KaplanMeierFitter
        Objet ajusté de Kaplan-Meier
        
    Example:
    --------
    >>> kmf = build_kaplan_meier(preprocessed_df, 'alzheimer', 'UK', 70)
    >>> print(f"Médiane de survie: {kmf.median_survival_time_} jours")
    """
    
    # Extraire le DataFrame approprié selon le format d'entrée
    if isinstance(preprocessed_df, dict):
        # Format: preprocessed_df[disease][country_age]
        country_age_key = f"{country}_{age}"
        if disease not in preprocessed_df:
            raise ValueError(f"Maladie '{disease}' non trouvée dans preprocessed_df")
        if country_age_key not in preprocessed_df[disease]:
            raise ValueError(f"Combinaison '{country_age_key}' non trouvée pour la maladie '{disease}'")
        df = preprocessed_df[disease][country_age_key]
    else:
        # Format: DataFrame direct
        df = preprocessed_df
    
    # Vérifier que les colonnes nécessaires sont présentes
    required_cols = ['time', 'event']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
    
    # Extraire les données de survie
    # Pour Kaplan-Meier, on ne considère que les événements d'intérêt (maladie = 1)
    # Les décès (event = 2) sont traités comme des censures
    durations = df['time'].copy()
    events = (df['event'] == 1).astype(int)  # 1 si maladie, 0 sinon
    
    # Filtrer les temps négatifs ou nuls (patients avec événements avant t=0)
    valid_mask = durations > 0
    n_invalid = (~valid_mask).sum()
    
    if n_invalid > 0:
        print(f"⚠️  Exclusion de {n_invalid} patients avec temps ≤ 0 (événements avant baseline)")
        durations = durations[valid_mask]
        events = events[valid_mask]
    else:
        print(f"✅ Aucun patient exclu (tous les temps > 0)")  # Message informatif
    
    # Vérifier qu'on a des données valides
    if len(durations) == 0:
        raise ValueError("Aucune donnée valide trouvée après filtrage")
    
    # Convertir les temps selon l'unité demandée
    time_conversion_factor = 365.25 if time_unit == 'years' else 1.0
    durations_display = durations / time_conversion_factor
    time_unit_label = 'années' if time_unit == 'years' else 'jours'
    
    # Créer et ajuster le modèle Kaplan-Meier (toujours avec les jours pour lifelines)
    kmf = KaplanMeierFitter(alpha=alpha)
    kmf.fit(durations, events, label=f'Apparition de {disease}')
    
    if show_plot:
        # Créer le graphique avec conversion temporelle
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Pour l'affichage, on doit convertir manuellement car lifelines travaille en jours
        if time_unit == 'years':
            # Créer les données converties pour l'affichage
            timeline_display = kmf.timeline / time_conversion_factor
            survival_func = kmf.survival_function_.copy()
            survival_func.index = survival_func.index / time_conversion_factor
            confidence_int = kmf.confidence_interval_.copy()
            confidence_int.index = confidence_int.index / time_conversion_factor
            
            # Plot manuel avec les données converties
            ax.plot(survival_func.index, survival_func.iloc[:, 0], 
                   label=f'Apparition de {disease}', linewidth=2)
            
            # Intervalles de confiance
            ax.fill_between(confidence_int.index, 
                          confidence_int.iloc[:, 0], 
                          confidence_int.iloc[:, 1], 
                          alpha=0.3)
        else:
            # Utiliser le plot standard de lifelines
            kmf.plot_survival_function(ax=ax, ci_show=True)
        
        # Personnaliser le graphique
        if title is None:
            title = f'Courbe de Kaplan-Meier - Apparition de {disease}\nPopulation: {country}_{age}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Temps ({time_unit_label})', fontsize=12)
        ax.set_ylabel('Probabilité de survie sans maladie', fontsize=12)
        
        # Définir les limites de l'axe x 
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            # Par défaut, commencer à 0 et aller jusqu'au max des temps convertis
            max_time_display = durations.max() / time_conversion_factor
            ax.set_xlim(0, max_time_display * 1.05)  # Ajouter 5% de marge
        
        # Améliorer l'apparence
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Ajouter des informations statistiques
        median_time = kmf.median_survival_time_
        if not pd.isna(median_time):
            median_time_display = median_time / time_conversion_factor
            unit_short = 'ans' if time_unit == 'years' else 'j'
            ax.axvline(x=median_time_display, color='red', linestyle='--', alpha=0.7, 
                      label=f'Médiane: {median_time_display:.1f} {unit_short}')
        
        # Ajouter le nombre à risque (après filtrage)
        n_total = len(durations)
        n_events = events.sum()
        n_censored = n_total - n_events
        n_original = len(df)
        n_excluded = n_original - n_total
        
        info_text = f'N analysé: {n_total}\nExclus: {n_excluded}'
        info_text += f'\nÉvénements: {n_events}\nCensurés: {n_censored}'
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='bottom', fontsize=10)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Afficher quelques statistiques
        print(f"\n=== Statistiques de survie pour {disease} - {country}_{age} ===")
        print(f"Nombre de sujets analysés: {n_total}")
        print(f"Nombre de sujets exclus (temps ≤ 0): {n_excluded}")
        print(f"Nombre d'événements (maladie): {n_events}")
        print(f"Nombre de censures: {n_censored}")
        
        if not pd.isna(median_time):
            median_display = median_time / time_conversion_factor
            if time_unit == 'years':
                print(f"Temps médian jusqu'à l'événement: {median_display:.1f} années")
            else:
                print(f"Temps médian jusqu'à l'événement: {median_display:.1f} jours ({median_time/365.25:.1f} années)")
        else:
            print("Temps médian jusqu'à l'événement: non atteint")
            
        # Probabilité de survie à différents moments
        if time_unit == 'years':
            time_points_years = [1, 2, 5, 10]  # années
            time_points_days = [int(y * 365.25) for y in time_points_years]
            print("\nProbabilités de survie sans maladie:")
            for years, t_days in zip(time_points_years, time_points_days):
                if t_days <= durations.max():
                    prob = kmf.predict(t_days)
                    print(f"  À {years} an(s): {prob:.3f} ({prob*100:.1f}%)")
        else:
            time_points = [365, 2*365, 5*365]  # 1, 2, 5 ans en jours
            print("\nProbabilités de survie sans maladie:")
            for t in time_points:
                if t <= durations.max():
                    prob = kmf.predict(t)
                    years = t / 365.25
                    print(f"  À {years:.0f} an(s) ({t} jours): {prob:.3f} ({prob*100:.1f}%)")
    
    return kmf


def compare_kaplan_meier(preprocessed_df, disease: str, 
                        country: str, age: int,
                        group_column: str, group_labels: Optional[dict] = None,
                        alpha: float = 0.05, title: Optional[str] = None,
                        xlim: Optional[Tuple[float, float]] = None,
                        show_plot: bool = True,
                        time_unit: str = 'years') -> dict:
    """
    Comparer les courbes de Kaplan-Meier entre différents groupes.
    
    Parameters:
    -----------
    preprocessed_df : dict ou pd.DataFrame
        Si dict: dictionnaire structuré preprocessed_df[disease][country_age] contenant les DataFrames
        Si pd.DataFrame: DataFrame préprocessé
        
    disease : str
        Nom de la maladie
        
    country : str
        Pays ('FR' ou 'UK')
        
    age : int
        Âge de la cohorte (65 ou 70)
        
    group_column : str
        Nom de la colonne pour grouper les sujets
        
    group_labels : dict, optional
        Dictionnaire pour renommer les groupes {valeur_originale: nouveau_label}
        
    alpha : float, default=0.05
        Niveau de confiance
        
    title : str, optional
        Titre personnalisé
        
    xlim : tuple, optional
        Limites de l'axe x
        
    show_plot : bool, default=True
        Afficher le graphique
        
    time_unit : str, default='years'
        Unité de temps pour l'affichage ('years' ou 'days')
        
    Returns:
    --------
    dict
        Dictionnaire contenant les objets KaplanMeierFitter pour chaque groupe
        
    Example:
    --------
    >>> kmf_groups = compare_kaplan_meier(preprocessed_df, 'alzheimer', 'UK', 70, 'is female', 
    ...                                  group_labels={0: 'Hommes', 1: 'Femmes'})
    """
    
    # Extraire le DataFrame approprié selon le format d'entrée
    if isinstance(preprocessed_df, dict):
        # Format: preprocessed_df[disease][country_age]
        country_age_key = f"{country}_{age}"
        if disease not in preprocessed_df:
            raise ValueError(f"Maladie '{disease}' non trouvée dans preprocessed_df")
        if country_age_key not in preprocessed_df[disease]:
            raise ValueError(f"Combinaison '{country_age_key}' non trouvée pour la maladie '{disease}'")
        df = preprocessed_df[disease][country_age_key]
    else:
        # Format: DataFrame direct
        df = preprocessed_df
    
    # Vérifier que la colonne de groupe existe
    if group_column not in df.columns:
        raise ValueError(f"Colonne '{group_column}' non trouvée dans le DataFrame")
    
    # Obtenir les groupes uniques
    unique_groups = df[group_column].dropna().unique()
    
    if len(unique_groups) < 2:
        raise ValueError(f"Il faut au moins 2 groupes dans la colonne '{group_column}'")
    
    # Créer un dictionnaire pour stocker les modèles
    kmf_groups = {}
    
    # Paramètres de conversion temporelle
    time_conversion_factor = 365.25 if time_unit == 'years' else 1.0
    time_unit_label = 'années' if time_unit == 'years' else 'jours'
    
    if show_plot:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    
    # Ajuster un modèle pour chaque groupe
    for group in unique_groups:
        group_data = df[df[group_column] == group].copy()
        
        if len(group_data) == 0:
            continue
            
        # Extraire les données de survie pour ce groupe
        durations = group_data['time'].copy()
        events = (group_data['event'] == 1).astype(int)
        
        # Filtrer les temps négatifs ou nuls pour ce groupe
        valid_mask = durations > 0
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            print(f"⚠️  Groupe {group}: exclusion de {n_invalid} patients avec temps ≤ 0")
            durations = durations[valid_mask]
            events = events[valid_mask]
        
        # Déterminer le label du groupe
        if group_labels and group in group_labels:
            label = group_labels[group]
        else:
            label = f"{group_column} = {group}"
        
        # Ajuster le modèle Kaplan-Meier
        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit(durations, events, label=label)
        kmf_groups[group] = kmf
        
        if show_plot:
            # Tracer la courbe pour ce groupe avec conversion temporelle
            if time_unit == 'years':
                # Plot manuel avec conversion
                survival_func = kmf.survival_function_.copy()
                survival_func.index = survival_func.index / time_conversion_factor
                confidence_int = kmf.confidence_interval_.copy() 
                confidence_int.index = confidence_int.index / time_conversion_factor
                
                ax.plot(survival_func.index, survival_func.iloc[:, 0], 
                       label=label, linewidth=2)
                ax.fill_between(confidence_int.index,
                              confidence_int.iloc[:, 0],
                              confidence_int.iloc[:, 1],
                              alpha=0.2)
            else:
                # Plot standard
                kmf.plot_survival_function(ax=ax, ci_show=True, label=label)
    
    if show_plot:
        # Personnaliser le graphique
        if title is None:
            title = f'Comparaison des courbes de Kaplan-Meier - {disease}\npar {group_column} - Population: {country}_{age}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Temps ({time_unit_label})', fontsize=12)
        ax.set_ylabel('Probabilité de survie sans maladie', fontsize=12)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            # Par défaut, commencer à 0 et aller jusqu'au max des temps de tous les groupes
            max_time = df['time'].max() / time_conversion_factor
            ax.set_xlim(0, max_time * 1.05)  # Ajouter 5% de marge
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques pour chaque groupe
        print(f"\n=== Comparaison des groupes pour {disease} - {country}_{age} ===")
        for group, kmf in kmf_groups.items():
            # Données originales du groupe
            group_data_orig = df[df[group_column] == group]
            # Données après filtrage (temps > 0)
            group_data_valid = group_data_orig[group_data_orig['time'] > 0]
            
            n_original = len(group_data_orig)
            n_analyzed = len(group_data_valid)
            n_excluded = n_original - n_analyzed
            n_events = (group_data_valid['event'] == 1).sum()
            
            label = group_labels.get(group, group) if group_labels else group
            print(f"\nGroupe '{label}':")
            print(f"  N analysé: {n_analyzed}")
            print(f"  N exclu (temps ≤ 0): {n_excluded}")
            print(f"  Événements: {n_events}")
            
            median_time = kmf.median_survival_time_
            if not pd.isna(median_time):
                median_display = median_time / time_conversion_factor
                if time_unit == 'years':
                    print(f"  Temps médian: {median_display:.1f} années")
                else:
                    print(f"  Temps médian: {median_display:.1f} jours ({median_time/365.25:.1f} années)")
            else:
                print("  Temps médian: non atteint")
    
    return kmf_groups


def compare_kaplan_meier_by_age(preprocessed_df, disease: str, 
                               country: str, ages: list = [65, 70],
                               alpha: float = 0.05, title: Optional[str] = None,
                               xlim: Optional[Tuple[float, float]] = None,
                               show_plot: bool = True,
                               time_unit: str = 'years') -> dict:
    """
    Comparer les courbes de Kaplan-Meier entre différentes cohortes d'âge.
    
    Parameters:
    -----------
    preprocessed_df : dict ou pd.DataFrame
        Si dict: dictionnaire structuré preprocessed_df[disease][country_age] contenant les DataFrames
        Si pd.DataFrame: DataFrame préprocessé
        
    disease : str
        Nom de la maladie
        
    country : str
        Pays ('FR' ou 'UK')
        
    ages : list, default=[65, 70]
        Liste des âges des cohortes à comparer
        
    alpha : float, default=0.05
        Niveau de confiance
        
    title : str, optional
        Titre personnalisé
        
    xlim : tuple, optional
        Limites de l'axe x
        
    show_plot : bool, default=True
        Afficher le graphique
        
    time_unit : str, default='years'
        Unité de temps pour l'affichage ('years' ou 'days')
        
    Returns:
    --------
    dict
        Dictionnaire contenant les objets KaplanMeierFitter pour chaque âge
        
    Example:
    --------
    >>> kmf_ages = compare_kaplan_meier_by_age(preprocessed_df, 'alzheimer', 'UK', [65, 70])
    """
    
    # Vérifier que nous avons au moins 2 âges
    if len(ages) < 2:
        raise ValueError("Il faut au moins 2 âges différents pour faire une comparaison")
    
    # Créer un dictionnaire pour stocker les modèles
    kmf_ages = {}
    
    # Paramètres de conversion temporelle
    time_conversion_factor = 365.25 if time_unit == 'years' else 1.0
    time_unit_label = 'années' if time_unit == 'years' else 'jours'
    
    if show_plot:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Couleurs distinctes
    
    # Ajuster un modèle pour chaque âge
    for i, age in enumerate(ages):
        # Extraire le DataFrame approprié selon le format d'entrée
        if isinstance(preprocessed_df, dict):
            # Format: preprocessed_df[disease][country_age]
            country_age_key = f"{country}_{age}"
            if disease not in preprocessed_df:
                print(f"⚠️  Maladie '{disease}' non trouvée - âge {age} ignoré")
                continue
            if country_age_key not in preprocessed_df[disease]:
                print(f"⚠️  Combinaison '{country_age_key}' non trouvée pour '{disease}' - âge {age} ignoré")
                continue
            df_age = preprocessed_df[disease][country_age_key]
        else:
            # Si DataFrame direct, on ne peut pas séparer par âge
            raise ValueError("Pour comparer par âge, preprocessed_df doit être un dictionnaire avec [disease][country_age]")
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_cols = ['time', 'event']
        missing_cols = [col for col in required_cols if col not in df_age.columns]
        if missing_cols:
            print(f"⚠️  Colonnes manquantes pour {country}_{age}: {missing_cols} - âge ignoré")
            continue
            
        if len(df_age) == 0:
            print(f"⚠️  Aucune donnée pour {country}_{age} - âge ignoré")
            continue
            
        # Extraire les données de survie pour cet âge
        durations = df_age['time'].copy()
        events = (df_age['event'] == 1).astype(int)
        
        # Filtrer les temps négatifs ou nuls pour cette cohorte
        valid_mask = durations > 0
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            print(f"⚠️  Cohorte {age} ans: exclusion de {n_invalid} patients avec temps ≤ 0")
            durations = durations[valid_mask]
            events = events[valid_mask]
        
        # Vérifier qu'il reste des données
        if len(durations) == 0:
            print(f"⚠️  Aucune donnée valide pour la cohorte {age} ans après filtrage")
            continue
        
        # Label pour cette cohorte
        label = f'{country} {age} ans'
        
        # Ajuster le modèle Kaplan-Meier
        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit(durations, events, label=label)
        kmf_ages[age] = kmf
        
        if show_plot:
            # Tracer la courbe pour cette cohorte avec conversion temporelle
            color = colors[i % len(colors)]
            if time_unit == 'years':
                # Plot manuel avec conversion
                survival_func = kmf.survival_function_.copy()
                survival_func.index = survival_func.index / time_conversion_factor
                confidence_int = kmf.confidence_interval_.copy() 
                confidence_int.index = confidence_int.index / time_conversion_factor
                
                ax.plot(survival_func.index, survival_func.iloc[:, 0], 
                       label=label, linewidth=2.5, color=color)
                ax.fill_between(confidence_int.index,
                              confidence_int.iloc[:, 0],
                              confidence_int.iloc[:, 1],
                              alpha=0.2, color=color)
            else:
                # Plot standard
                kmf.plot_survival_function(ax=ax, ci_show=True, label=label, color=color)
    
    if len(kmf_ages) == 0:
        raise ValueError("Aucune cohorte d'âge valide trouvée")
    
    if show_plot:
        # Personnaliser le graphique
        if title is None:
            ages_str = " vs ".join([f"{age} ans" for age in sorted(kmf_ages.keys())])
            title = f'Comparaison par âge - {disease}\n{country}: {ages_str}'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Temps ({time_unit_label})', fontsize=12)
        ax.set_ylabel('Probabilité de survie sans maladie', fontsize=12)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            # Par défaut, commencer à 0 et aller jusqu'au max des temps de toutes les cohortes
            all_dfs = []
            for age in kmf_ages.keys():
                country_age_key = f"{country}_{age}"
                if isinstance(preprocessed_df, dict) and disease in preprocessed_df:
                    if country_age_key in preprocessed_df[disease]:
                        all_dfs.append(preprocessed_df[disease][country_age_key])
            
            if all_dfs:
                max_time = max(df['time'].max() for df in all_dfs) / time_conversion_factor
                ax.set_xlim(0, max_time * 1.05)  # Ajouter 5% de marge
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques pour chaque cohorte d'âge
        print(f"\n=== Comparaison des cohortes d'âge pour {disease} - {country} ===")
        for age in sorted(kmf_ages.keys()):
            kmf = kmf_ages[age]
            
            # Données originales et filtrées
            country_age_key = f"{country}_{age}"
            if isinstance(preprocessed_df, dict) and disease in preprocessed_df:
                if country_age_key in preprocessed_df[disease]:
                    age_data_orig = preprocessed_df[disease][country_age_key]
                    age_data_valid = age_data_orig[age_data_orig['time'] > 0]
                    
                    n_original = len(age_data_orig)
                    n_analyzed = len(age_data_valid)
                    n_excluded = n_original - n_analyzed
                    n_events = (age_data_valid['event'] == 1).sum()
                    
                    print(f"\nCohorte {age} ans:")
                    print(f"  N analysé: {n_analyzed}")
                    print(f"  N exclu (temps ≤ 0): {n_excluded}")
                    print(f"  Événements: {n_events}")
                    
                    median_time = kmf.median_survival_time_
                    if not pd.isna(median_time):
                        median_display = median_time / time_conversion_factor
                        if time_unit == 'years':
                            print(f"  Temps médian: {median_display:.1f} années")
                        else:
                            print(f"  Temps médian: {median_display:.1f} jours ({median_time/365.25:.1f} années)")
                    else:
                        print("  Temps médian: non atteint")
        
        # Comparaison des médianes
        print("\n--- Comparaison des temps médians ---")
        medians = {}
        for age, kmf in kmf_ages.items():
            median_time = kmf.median_survival_time_
            if not pd.isna(median_time):
                medians[age] = median_time / time_conversion_factor
        
        if len(medians) > 1:
            ages_sorted = sorted(medians.keys())
            unit_short = 'ans' if time_unit == 'years' else 'jours'
            
            for i in range(len(ages_sorted)-1):
                age1, age2 = ages_sorted[i], ages_sorted[i+1]
                if age1 in medians and age2 in medians:
                    diff = medians[age2] - medians[age1]
                    direction = "plus tard" if diff > 0 else "plus tôt"
                    print(f"Cohorte {age2} ans vs {age1} ans: {abs(diff):.1f} {unit_short} {direction}")
    
    return kmf_ages