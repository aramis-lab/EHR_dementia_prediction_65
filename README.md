# Data-driven risk algorithms from primary care for early dementia detection and prevention trial enrichment

## Overview

This repository contains the code and analysis for predicting dementia and Alzheimer's disease onset using medication prescription patterns from electronic health records (EHR). The study leverages data from UK and French primary care databases to develop age-specific prediction models that maintain discriminatory power within homogeneous age cohorts.

## Key Features

- **Multi-country validation**: Models trained on UK data and validated on French data
- **Age-specific predictions**: Focused on 65 and 70-year-old cohorts to avoid age-bias
- **Multiple algorithms comparison**: Logistic Regression, Random Forest, SVM, and Neural Networks
- **Clinical utility metrics**: Including precision of top 1% patients and enrichment factors
- **Temporal analysis**: 2, 5, and 10-year prediction horizons
- **Comprehensive evaluation**: ROC AUC, Brier scores, detection rates, and calibration metrics

## Workflow

### 1. Creation of datasets (`1. Creation of datasets.ipynb`)
- **Data loading and preprocessing** from UK and French primary care databases
- **Feature engineering** from medication prescriptions (ATC codes)
- **Outcome definition** for dementia and Alzheimer's disease
- **Baseline characteristics analysis** including demographics and medication patterns

### 2. Models (`2. Models.ipynb`)
- **Data splits** between datasets for feature selection and for training / validation machin learning algorithms
- **Fine-Gray competing risk models** for medication associations
- **Statistical significance testing** with Bonferroni correction

### 3. Prediction (`3. Prediction.ipynb`)
- **Machine learning algorithm comparison** (Logistic Regression, Random Forest, SVM, Neural Networks)
- **Performance metrics calculation** (ROC AUC, Brier scores, detection rates)
- **Clinical utility assessment** (precision of top 1%, enrichment factors)
- **ROC curve visualization** and multi-algorithm comparison
- **Temporal analysis** across 2, 5, and 10-year horizons

Execute the notebooks in order:

1. Start with data creation: Open 1. Creation of datasets.ipynb
- Process raw healthcare data
- Generate baseline characteristics tables
- Create analysis-ready datasets
2. Statistical modeling: Run 2. Models.ipynb
- Identify significant medication associations
- Calculate hazard ratios with competing risks
- Generate statistical results tables
3. Prediction analysis: Execute 3. Prediction.ipynb
- Compare machine learning algorithms
- Evaluate clinical utility metrics
- Generate performance tables and visualizations

##  Input Data Format

The analysis expects healthcare data with:

- Patient demographics: Age, sex, baseline year
- Medication prescriptions: ATC codes with prescription dates
- Disease outcomes: ICD codes for dementia/Alzheimer's with diagnosis dates
- Follow-up information: Death dates, last contact dates

##  Dependencies

- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- lifelines>=0.27.0
- jupyter>=1.0.0
- scipy>=1.7.0
