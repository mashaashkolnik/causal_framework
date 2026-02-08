# Day-to-day dietary variation shapes overnight sleep physiology  
## A target-trial emulation in ~4.8 thousand person-nights

This repository contains all analysis code used to produce the results, figures, and robustness checks for the manuscript:

**Day-to-day dietary variation shapes overnight sleep physiology: a target-trial emulation in 4.8 thousand person-nights**

---

## Overview

We analyze ~4,800 person-nights from the **Human Phenotype Project (HPP)** combining:
- Time-stamped dietary logs (composition, quality, micronutrients, meal timing)
- Objective multi-stage sleep recordings (WatchPAT)

The analytic design emulates a **day-level target trial**, estimating the effect of realistic dietary contrasts (median-based high vs. low exposure days) on next-night sleep outcomes.

### Core methodological components
- Machine-learning propensity score estimation (CatBoost)
- Platt calibration of propensity scores
- Overlap restriction & quantile trimming
- Stabilized inverse-probability weighting (Hájek normalization)
- Nonparametric bootstrap uncertainty estimation (1,000 replicates)
- Balance diagnostics (ASMD with prespecified thresholds)
- Negative and positive controls

Outcomes span sleep duration, sleep-stage composition (deep/REM/light), continuity metrics, and nocturnal autonomic physiology (mean heart rate).

---

## Repository structure

```text
CAUSAL_FRAMEWORK/
│
├── scripts/                      
│   ├── helpers.py                 # Experiment runner, utilities
│   ├── ipw.py                     # Trimming, weighting, bootstrap ATEs
│   ├── matching.py                # Propensity score matching
│   ├── propensity.py              # Propensity score estimation
│   └── plot.py                    # Figure generation
│
├── variables/                     
│   ├── configs.py                 # Global experiment configuration
│   ├── labels.py                  # Human-readable variable labels
│   └── variables.py               # Exposure, outcome, confounder lists
│
├── baseline_characteristics.ipynb # Descriptive cohort statistics
├── causal_engine.ipynb            # Main causal pipeline
├── matching.ipynb                 # Supplementary analysis
├── results_aggregation.ipynb      # Result aggregation for the manuscript
│
├── manuscript/                    # Manuscript assets
├── results/                       # Outputs
├── results_matching/              # Supplementary outputs
│
├── LICENSE
└── README.md
