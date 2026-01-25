# Day-to-day dietary variation shapes overnight sleep physiology: a target-trial emulation in 4.8 thousand person-nights

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## Overview
This repository contains all code used to generate the results, figures, and supplementary analyses for our study on **effects of daily nutritional variation on next-night sleep physiology**.  
The workflow implements a modern target-trial emulation framework, including:

- Machine-learning propensity score estimation (CatBoost)
- Overlap trimming
- Stabilized inverse-probability weighting (Hájek normalization)
- Bootstrap uncertainty estimation
- Diagnostics (ASMD balance, overlap, PS calibration)
- Effect estimation across 15 objective sleep outcomes

All analyses were performed on ~4,800 person-nights from the **Human Phenotype Project (HPP)**.

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## Usage


## 🗂 Repository Structure  

```markdown
CAUSAL_FRAMEWORK/
│
├── scripts/                       # Experiment outputs (plots, logs, dataframes)
│   ├── helpers.py                 # run_experiment(), plotting utilities
│   ├── ipw.py                     # IPW trimming, weighting, bootstrap ATE
│   ├── matching.py                # Propensity score estimation & SHAP
│   └── plot.py                    # Configuration: exposures, outcomes, confounders
│
├── variables/                     # Core causal framework code
│   ├── configs.py                 # run_experiment(), plotting utilities
│   ├── labels.py                  # IPW trimming, weighting, bootstrap ATE
│   └── variables.py               # Configuration: exposures, outcomes, confounders
│
├── paper_files/                   # (optional) additional storage
├── results/                       # (optional) additional storage
├── results_matching/              # (optional) additional storage
|
├── baseline_characteristics.ipynb # Notebook template for running experiments
├── causal_engine.ipynb            # Notebook template for running experiments
├── matching.ipynb                 # Notebook template for running experiments
├── results_aggregation.ipynb      # Notebook template for running experiments
│
├── LICENSE
└── README.md
```

<div style="border-bottom:1px solid #ccc; margin:20px 0;"></div>

## Setup Guide
 

