import json
from dataclasses import dataclass, replace
import numpy as np


DATAFRAME_PATH = "data/dataframe.csv"
FEATURES_PATH = "data/features.json"
SAVE_CHARTS = True
index = ["RegistrationCode", "id"]

config = {
    "n_iter": 2000,
    "n_features": 10,
    "calibration": "calibrated_scores_logreg",
    "q": 0.01,
    "t": "treated",
    "stabilize": True,
    "clip": (1, 99),
    "alpha": 0.01,
}

sleep_targets = [
    # Overall Food-Sensitive Sleep Quality Index (weighted combination of continuity, architecture, oxygenation, autonomic)
    # "composite_target_day",
    # Measures how consolidated sleep was (efficiency, latency, WASO, number of wakes)
    "composite_sleep_quality_target_day",
    "sleep_quality_continuity_target_day",
    "sleep_quality_architecture_target_day",
    # "sleep_quality_oxygenation_target_day",
    # "sleep_quality_autonomic_target_day",
    # Most diet-sensitive (directly and strongly affected)
    "sleep_latency_minutes_target_day",
    "sleep_efficiency_target_day",
    "total_wake_time_after_sleep_onset_minutes_target_day",
    "total_arousal_sleep_time_target_day",
    "number_of_wakes_target_day",
    # Moderately affected (indirectly or via longer-term diet patterns)
    "percent_of_deep_sleep_time_target_day",
    "percent_of_rem_sleep_time_target_day",
    "percent_of_light_sleep_time_target_day",
    "total_deep_sleep_time_minutes_target_day",
    "total_rem_sleep_time_minutes_target_day",
    # "neurokit_hrv_frequency_hf_during_night_target_day",
    "total_sleep_time_minutes_target_day",
    # Weakly affected (mostly secondary to metabolic, weight, or cardiometabolic changes)
    "heart_rate_mean_during_sleep_target_day",
    # "hypoxic_burden_target_day",
    # "odi_target_day",
    # "snore_db_mean_target_day",
    # "rem_latency_target_day",
    # "number_of_transitions_rem_to_wake_target_day",
    # "saturation_min_value_target_day",
]

negative_targets = [
    # Signal quality score for SpO₂ measurements during sleep (higher = more reliable data)
    # "quality_score_spo2_target_day",
    # Signal quality score for heart rate measurements during sleep (higher = more reliable data)
    "quality_score_heart_rate_target_day",
    # Data quality score for HRV signal collected during wakefulness (higher = cleaner, more reliable signal)
    "neurokit_hrv_quality_score_during_wake_target_day",
    # Percentage of total sleep time spent lying on the back (supine position)
    "percent_of_supine_sleep_target_day",
    # Total number of minutes spent sleeping on the left side
    "total_left_sleep_time_minutes_target_day",
    # Sleep Hour
    # "sleep_hour_target_day",
    # "total_energy_kcal_target_day",
]

features_id = [
    "age",
    "gender",
    "weight",
    "bmi",
]


structural_confounders = [
    "age",
    "gender",
    "bmi",
    # "weight",
]

direct_confounders = [
    "composite_sleep_quality",
    "sleep_quality_continuity",
    "sleep_quality_architecture",
    # Most diet-sensitive (directly and strongly affected)
    "sleep_latency_minutes",
    "sleep_efficiency",
    "total_wake_time_after_sleep_onset_minutes",
    "total_arousal_sleep_time",
    "number_of_wakes",
    # Moderately affected (indirectly or via longer-term diet patterns)
    "percent_of_deep_sleep_time",
    "percent_of_rem_sleep_time",
    "percent_of_light_sleep_time",
    "total_deep_sleep_time_minutes",
    "total_rem_sleep_time_minutes",
    # "neurokit_hrv_frequency_hf_during_night",
    "total_sleep_time_minutes",
    # Weakly affected (mostly secondary to metabolic, weight, or cardiometabolic changes)
    "heart_rate_mean_during_sleep",
]

partial_confounders = [
    "total_energy_kcal",
    "sleep_hour",
]

features = json.loads(open(FEATURES_PATH).read())
numerical, categorical, target = (
    features["numerical"],
    features["categorical"],
    "is_treated",
)

variable_config = {
    "sleep_targets": sleep_targets,
    "features_id": features_id,
    "negative_targets": negative_targets,
    "structural_confounders": structural_confounders,
    "direct_confounders": direct_confounders,
    "partial_confounders": partial_confounders,
    "numerical": numerical,
    "categorical": categorical,
    "target": target,
    "confounders": structural_confounders + direct_confounders + partial_confounders,
}

# ================================================================
# Configuration dataclass (clean, immutable)
# ================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    n_iter: int = 2000
    n_features: int = 10
    calibration: str = "calibrated_scores_logreg"
    q: float = 0.01
    t: str = "treated"
    stabilize: bool = True
    clip: tuple = (1, 99)
    alpha: float = 0.01
    method: str = "gb"


# Base configuration
BASE_CONFIG = ExperimentConfig()

EXPERIMENT_ID = 1
