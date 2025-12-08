# Daily dietary choices acutely shape next-night sleep architecture: a target-trial emulation in a 6,000 person-nights cohort

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

---

## 🗂 Repository Structure  

```markdown
project_root/
│
├── src/                         # Main source code
│   ├── data_utils.py            # Data loading & preprocessing helpers
│   ├── models.py                # Model definitions or ML workflows
│   ├── analysis.py              # Core analysis functions
│   ├── plotting.py              # Plotting and figure generation
│   └── __init__.py
│
├── notebooks/                   # Jupyter notebooks for step-by-step workflow
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_effect_estimation.ipynb
│   └── 04_figure_generation.ipynb
│
├── configs/                     # Configuration files for experiments
│   ├── main_config.yaml
│   └── hyperparameters.yaml
│
├── results/                     # Outputs: figures, tables, logs
│   ├── figures/
│   ├── tables/
│   └── diagnostics/
│
├── data/                        # Raw or processed data (usually ignored in .gitignore)
│   └── README.md                # Instructions for obtaining data
│
├── environment.yml              # Conda environment configuration
├── requirements.txt             # pip dependencies
├── LICENSE
└── README.md
```

---



