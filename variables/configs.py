import json
from dataclasses import dataclass, replace
import numpy as np

# sleep limiting
final_configs = {
    # ---------------- MACROS ----------------
    "prot_pct_target_day" : {'q' : 0.0, 'clip' : (7, 93)}, #{'q' : 0.0125, 'clip' : (1.5, 98.5)},
    "fat_pct_target_day" : {'q' : 0.001, 'clip' : (5.0, 95.0)},
    "carb_pct_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    # ---------------- NUTRIENTS ----------------
    "omega3_total_g_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    "magnesium_mg_target_day" : {'q':0.01, 'clip' : None}, 
    "calcium_mg_target_day" : {'q' : 0.0, 'clip' : (5, 95)}, 
    "vitamin_d_ug_target_day" : {'q' : 0.015, 'clip' : None},
    "vitamin_b6_mg_target_day" :{'q' : 0.0, 'clip' : (5.0, 95.0)}, # q=0.001, clip=(5.0, 95.0) {'q' : 0.01, 'clip' : (2.5, 97.5)}
    "vitamin_b12_ug_target_day" : {'q' : 0.01, 'clip' : (5.0, 95.0)},
    "zinc_mg_target_day" : {'q' : 0.01, 'clip' : (1.0, 99.0)},
    "folate_target_day" : {'q' : 0.025, 'clip' : (1.5, 98.5)},
    "vitamin_c_target_day" : {'q' : 0.0, 'clip' : (7, 93)}, #{'q' : 0.01, 'clip' : (2.5, 97.5)},
    "vitamin_e_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    # ---------------- TIMING ----------------
    "hours_to_sleep_target_day" : {'q' : 0.015, 'clip' : (3.5, 96.5)}, #{'q' : 0.01, 'clip' : (3.5, 96.5)},
    "eating_window_h_target_day" : {'q' : 0.0125, 'clip' : (1.5, 98.5)}, # {'q' : 0.01, 'clip' : (1.5, 98.5)},
    "night_calories_pct_target_day" : {'q' : 0.025, 'clip' : (5.0, 95.0)}, #q=0.025, clip=(5.0, 95.0)
    # ---------------- DIET COMPOSITION ----------------
    "fiber_g_target_day" : {'q' : 0.0, 'clip' : (3.5, 96.5)},
    "unique_plant_based_foods_count_target_day" : {'q' : 0.001, 'clip' : (3, 97)},
    "sat_fat_g_target_day" : {'q' : 0.0175, 'clip' : (1.5, 98.5)},
    "whole_food_categories_ratio_target_day" : {'q' : 0.01, 'clip' : None},
    "plant_based_whole_foods_ratio_target_day" : {'q' : 0.0, 'clip' : (3.5, 96.5)}, # q=0.001, clip=(3.5, 96.5)
    "whole_dairy_categories_ratio_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},  
    "processed_categories_ratio_target_day" : {'q' : 0.0125, 'clip' : (1.5, 98.5)},
    #"furits_and_veggies_energy_ratio_target_day" : {'q' : 0.0, 'clip' : (3.5, 96.5)}, 
    "furits_and_veggies_energy_ratio_target_day" : {'q' : 0.005, 'clip' : (5, 95)},  
    "animal_based_whole_foods_ratio_target_day" : {'q' : 0.0125, 'clip' : (2.5, 97.5)}, #{'q' : 0.025, 'clip' : (0.5, 99.5)}
    "meat_and_poultry_energy_ratio_target_day" : {'q':0.0, 'clip':(7.0, 93.0)}, #q=0.001, clip=(5.0, 95.0)
    # ---------------- DENSITY INDICES ----------------
    "fiber_density_energy_target_day" : {'q' : 0.0, 'clip' : (5, 95)},
    # ---------------- COFFEE ----------------
    "caffeine_late_mg_target_day" : {'q' : 0.015, 'clip' : (1, 99)},
}

# 2025-12-25T00:10:21,7691,meat_and_poultry_energy_ratio,"PASS_STRICT | with alpha=0.0, q=0.0, clip=(7.0, 93.0), cutoff=None, method=median	",100.0,100.0,100.0,100.0,100.0,20,,,,,,"{'exposure_baseline': 'meat_and_poultry_energy_ratio', 'alpha': 0.0, 'q': 0.0, 'clip_low': 7.0, 'clip_high': 93.0, 'stabilize': True, 'n_iter': 1000, 'quantile_cut': None, 'cutoff_values': None, 'rdi_values': None, 'method': 'median'}"
# 2025-12-24T23:33:07,7553,animal_based_whole_foods_ratio,"PASS | with alpha=0.0, q=0.025, clip=(0.5, 99.5), cutoff=None, method=median	",100.0,66.66666666666667,57.142857142857146,100.0,100.0,22,,age,sleep_efficiency;sleep_latency_minutes;total_sleep_time_minutes,,,"{'exposure_baseline': 'animal_based_whole_foods_ratio', 'alpha': 0.0, 'q': 0.025, 'clip_low': 0.5, 'clip_high': 99.5, 'stabilize': True, 'n_iter': 1000, 'quantile_cut': None, 'cutoff_values': None, 'rdi_values': None, 'method': 'median'}"
# 2025-12-24T23:12:46,7477,unique_plant_based_foods_count,"PASS_STRICT | with alpha=0.0, q=0.001, clip=(3.0, 97.0), cutoff=None, method=median	",100.0,100.0,85.71428571428571,100.0,100.0,21,,,heart_rate_mean_during_sleep,,,"{'exposure_baseline': 'unique_plant_based_foods_count', 'alpha': 0.0, 'q': 0.001, 'clip_low': 3.0, 'clip_high': 97.0, 'stabilize': True, 'n_iter': 1000, 'quantile_cut': None, 'cutoff_values': None, 'rdi_values': None, 'method': 'median'}"

# q=0.01, clip=(5.0, 95.0)