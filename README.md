# Daily dietary choices acutely shape next-night sleep architecture: a target-trial emulation in a 6,000 person-nights cohort

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## Overview
This repository contains all code used to generate the results, figures, and supplementary analyses for our study on **quasi-causal effects of daily nutritional variation on next-night sleep physiology**.  
The workflow implements a modern target-trial emulation framework, including:

- Machine-learning propensity score estimation (CatBoost)
- Overlap trimming
- Stabilized inverse-probability weighting (Hájek normalization)
- Bootstrap uncertainty estimation
- Diagnostics (ASMD balance, overlap, PS calibration)
- Effect estimation across 15 objective sleep outcomes

All analyses were performed on ~6,000 person-nights from the **Human Phenotype Project (HPP)**.

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## 🗂 Repository Structure  

```markdown
CAUSAL_FRAMEWORK/
│
├── catboost_info/                 # CatBoost metadata (auto-generated)
├── data/                          # Place your input dataset(s) here
│
├── experiment/                    # Experiment outputs (plots, logs, dataframes)
│   ├── results/
│   │   ├── charts/                # Auto-generated ATE plots
│   │   ├── dataframes/            # ASMD & ATE tables
│   │   └── experiment_summaries.csv
│
├── helpers/                       # Core causal framework code
│   ├── helpers.py                 # run_experiment(), plotting utilities
│   ├── ipw.py                     # IPW trimming, weighting, bootstrap ATE
│   ├── propensity.py              # Propensity score estimation & SHAP
│   └── variables.py               # Configuration: exposures, outcomes, confounders
│
├── outputs/                       # (optional) additional storage
├── template.ipynb                 # Notebook template for running experiments
│
├── LICENSE
└── README.md
```

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## Setup Guide


