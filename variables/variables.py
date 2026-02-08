import json
from dataclasses import dataclass

# ================================================================
# File paths
# ================================================================
DATAFRAME_PATH = "data/dec16_dataframe.csv"
FEATURES_PATH = "data/features_upd.json"

errorbar_folder_path = "results/charts"
log_csv_path = "results/log.csv"
df_folder_path = "results/dataframes"
result_plot_folder_path = "results/final_plot"

paper_tables_folder = "manuscript/tables"
paper_figures_folder = "manuscript/figures"

matching_folder_path_quantile = "results_matching/charts_quantile"
matching_df_folder_path = "results_matching/dataframes"

# ================================================================
# Index and targets
# ================================================================
index = ["RegistrationCode", "id"]

sleep_targets = [
    "total_sleep_time_minutes_target_day",
    "percent_of_deep_sleep_time_target_day",
    "percent_of_rem_sleep_time_target_day",
    "percent_of_light_sleep_time_target_day",
    "sleep_efficiency_target_day",
    "sleep_latency_minutes_target_day",
    "heart_rate_mean_during_sleep_target_day",
    "total_wake_time_after_sleep_onset_minutes_target_day",
]

negative_targets = [
    "quality_score_heart_rate_target_day",
    "percent_of_supine_sleep_target_day",
    "total_left_sleep_time_minutes_target_day",
]

# ================================================================
# Features and confounders
# ================================================================
features_id = ["age", "gender", "bmi"]

structural_confounders = ["age", "gender", "bmi"]

direct_confounders = [
    "percent_of_deep_sleep_time",
    "percent_of_rem_sleep_time",
    "percent_of_light_sleep_time",
    "heart_rate_mean_during_sleep",
    "sleep_efficiency",
    "total_sleep_time_minutes",
    "sleep_latency_minutes",
    "total_wake_time_after_sleep_onset_minutes",
    "number_of_wakes",
]

partial_confounders = ["total_energy_kcal", "sleep_hour"]

# ================================================================
# Load features
# ================================================================
with open(FEATURES_PATH) as f:
    features = json.load(f)

numerical = features["numerical"]
categorical = features["categorical"]
target = "is_treated"

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
# Experiment configuration
# ================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    n_iter: int = 1000
    n_features: int = 15
    calibration: str = "calibrated_scores_logreg"
    q: float = 0.01
    t: str = "treated"
    stabilize: bool = True
    clip: tuple = (1, 99)
    alpha: float = 0.0
    method: str = "median"
    limit: tuple = None
    quantile: float = 0.3

BASE_CONFIG = ExperimentConfig()

# ================================================================
# Exposures and their specific cutoffs/methods
# ================================================================
EXPOSURES = {
    # ---------------- PLANT-BASED ----------------
    "plant_based_whole_foods_ratio_target_day": {"cutoff": None, "method": "median"},
    "furits_and_veggies_energy_ratio_target_day": {"cutoff": [0.025, 0.3], "method": "median"},
    "whole_food_categories_ratio_target_day": {"cutoff": None, "method": "median"},
    "whole_dairy_categories_ratio_target_day": {"cutoff": None, "method": "median"},
    "meat_and_poultry_energy_ratio_target_day": {"cutoff": None, "method": "median"},
    "processed_categories_ratio_target_day": {"cutoff": None, "method": "median"},
    "animal_based_whole_foods_ratio_target_day": {"cutoff": [0.1, 0.6], "method": "median"},

    # ---------------- SATURATED FAT & FIBER ----------------
    "sat_fat_g_target_day": {"cutoff": [0, 120], "method": "gb"},
    "fiber_density_energy_target_day": {"cutoff": None, "method": "median"},
    "unique_plant_based_foods_count_target_day": {"cutoff": None, "method": "median"},

    # ---------------- MACROS ----------------
    "prot_pct_target_day": {"cutoff": [0.05, 0.35], "method": "median"},
    "fat_pct_target_day": {"cutoff": None, "method": "median"},
    "carb_pct_target_day": {"cutoff": None, "method": "median"},

    # ---------------- NUTRIENTS ----------------
    "magnesium_mg_target_day": {"cutoff": None, "method": "gb"},
    "omega3_total_g_target_day": {"cutoff": None, "method": "gb"},
    "vitamin_d_ug_target_day": {"cutoff": None, "method": "gb"},
    "vitamin_b6_mg_target_day": {"cutoff": None, "method": "gb"},
    "vitamin_b12_ug_target_day": {"cutoff": None, "method": "gb"},
    "calcium_mg_target_day": {"cutoff": [150, 1850], "method": "gb"},
    "zinc_mg_target_day": {"cutoff": [2.5, 25], "method": "gb"},
    "folate_target_day": {"cutoff": None, "method": "gb"},
    "vitamin_c_target_day": {"cutoff": None, "method": "gb"},
    "vitamin_e_target_day": {"cutoff": None, "method": "gb"},

    # ---------------- TIMING ----------------
    "hours_to_sleep_target_day": {"cutoff": [0.5, 10], "method": "median"},
    "eating_window_h_target_day": {"cutoff": [4, 18], "method": "median"},

    # ---------------- COFFEE ----------------
    "caffeine_late_mg_target_day": {"cutoff": None, "method": "median"},

    # ---------------- LATE NIGHT ----------------
    "night_calories_pct_target_day": {"cutoff": [5, 100], "method": "median"},
}
