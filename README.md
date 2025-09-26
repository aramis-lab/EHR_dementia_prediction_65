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

## Repository Structure

code_repo_prediction/
├── 1. Creation of datasets.ipynb       # Data processing, cleaning, and feature engineering
├── 2. Models.ipynb                     # Statistical analysis and medication associations
├── 3. Prediction.ipynb                 # Machine learning algorithms and performance evaluation
├── requirements.txt                    # Python dependencies
└── README.md                          # This file

## Workflow

### 1. Creation of datasets (`1. Creation of datasets.ipynb`)
- **Data loading and preprocessing** from UK and French primary care databases
- **Feature engineering** from medication prescriptions (ATC codes)
- **Outcome definition** for dementia and Alzheimer's disease
- **Baseline characteristics analysis** including demographics and medication patterns
- **Data splits** for training and validation (UK/French cohorts)

### 2. Models (`2. Models.ipynb`)
- **Fine-Gray competing risk models** for medication associations
- **Survival analysis** with time-to-event outcomes
- **Statistical significance testing** with Bonferroni correction
- **Hazard ratio calculations** for medication classes and subclasses
- **Identification of key predictive medications**

### 3. Prediction (`3. Prediction.ipynb`)
- **Machine learning algorithm comparison** (Logistic Regression, Random Forest, SVM, Neural Networks)
- **Cross-country validation** (UK training, French testing)
- **Performance metrics calculation** (ROC AUC, Brier scores, detection rates)
- **Clinical utility assessment** (precision of top 1%, enrichment factors)
- **ROC curve visualization** and multi-algorithm comparison
- **Temporal analysis** across 2, 5, and 10-year horizons

## Key Results

### Algorithm Performance Comparison

- **Logistic Regression** consistently outperformed complex algorithms across all scenarios
- For 2-year dementia prediction: AUROC 0.83 (95% CI 0.75-0.91)
- Superior detection rates: 54.6% of cases detected at 5% false-positive rate
- Better calibration with low Brier scores (0.159-0.226)

### Clinical Utility

- **Top 1% enrichment**: 46.2-fold enrichment for 2-year dementia prediction
- **Cross-country validation**: Consistent performance across UK and French populations
- **Temporal dynamics**: Performance declines expectedly over longer horizons

### Key Medication Associations (from Models notebook)

Strongest predictors identified through Fine-Gray competing risk models:
- **Psychoanaleptics/Antidepressants** (N06): HR 2.27 for dementia, 2.24 for Alzheimer's
- **Laxatives** (A06): HR 1.95 for dementia
- **Urological drugs** (G04): HR 1.66-1.95
- **Iron preparations** (B03): HR 1.86-2.07

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code_repo_prediction.git
cd code_repo_prediction

2. Install dependencies:
pip install -r requirements.txt

3. Set up Jupyter environment:
jupyter notebook

Usage

Running the Complete Analysis Pipeline

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

Key Functions Available

From Prediction Notebook

# Generate comprehensive performance table
performance_table = generate_performance_table(
    diseases=['all_dementias', 'alzheimer'],
    prediction_years=[2, 5, 10],
    age=65,
    include_charlson_bmi=True,
    n_bootstrap=1000
)

# Plot ROC curves for multiple time horizons
plot_roc_curves_multitime(
    country='UK',
    age=65,
    disease='all_dementias',
    prediction_years=[2, 5, 10]
)

# Compare multiple algorithms
multi_algo_table = generate_multi_algorithm_table(
    diseases=['all_dementias', 'alzheimer'],
    prediction_years=[2, 5, 10]
)

Data Requirements

Input Data Format

The analysis expects healthcare data with:

- Patient demographics: Age, sex, baseline year
- Medication prescriptions: ATC codes with prescription dates
- Disease outcomes: ICD codes for dementia/Alzheimer's with diagnosis dates
- Follow-up information: Death dates, last contact dates

Dependencies

pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
lifelines>=0.27.0
jupyter>=1.0.0
scipy>=1.7.0
