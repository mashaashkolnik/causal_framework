final_configs = {
    # ---------------- MACROS ----------------
    "prot_pct_target_day" : {'q' : 0.0, 'clip' : (7, 93)}, 
    "fat_pct_target_day" : {'q' : 0.001, 'clip' : (5.0, 95.0)},
    "carb_pct_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    # ---------------- NUTRIENTS ----------------
    "omega3_total_g_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    "magnesium_mg_target_day" : {'q':0.01, 'clip' : None}, 
    "calcium_mg_target_day" : {'q' : 0.0, 'clip' : (5, 95)}, 
    "vitamin_d_ug_target_day" : {'q' : 0.015, 'clip' : None},
    "vitamin_b6_mg_target_day" :{'q' : 0.0, 'clip' : (5.0, 95.0)},
    "vitamin_b12_ug_target_day" : {'q' : 0.01, 'clip' : (5.0, 95.0)},
    "zinc_mg_target_day" : {'q' : 0.01, 'clip' : (1.0, 99.0)},
    "folate_target_day" : {'q' : 0.025, 'clip' : (1.5, 98.5)},
    "vitamin_c_target_day" : {'q' : 0.0, 'clip' : (7, 93)},
    "vitamin_e_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},
    # ---------------- TIMING ----------------
    "hours_to_sleep_target_day" : {'q' : 0.015, 'clip' : (3.5, 96.5)},
    "eating_window_h_target_day" : {'q' : 0.0125, 'clip' : (1.5, 98.5)},
    "night_calories_pct_target_day" : {'q' : 0.025, 'clip' : (5.0, 95.0)},
    # ---------------- DIET COMPOSITION ----------------
    "fiber_g_target_day" : {'q' : 0.0, 'clip' : (3.5, 96.5)},
    "unique_plant_based_foods_count_target_day" : {'q' : 0.001, 'clip' : (3, 97)},
    "sat_fat_g_target_day" : {'q' : 0.0175, 'clip' : (1.5, 98.5)},
    "whole_food_categories_ratio_target_day" : {'q' : 0.01, 'clip' : None},
    "plant_based_whole_foods_ratio_target_day" : {'q' : 0.0, 'clip' : (3.5, 96.5)},
    "whole_dairy_categories_ratio_target_day" : {'q' : 0.01, 'clip' : (2.5, 97.5)},  
    "processed_categories_ratio_target_day" : {'q' : 0.0125, 'clip' : (1.5, 98.5)},
    "furits_and_veggies_energy_ratio_target_day" : {'q' : 0.005, 'clip' : (5, 95)},  
    "animal_based_whole_foods_ratio_target_day" : {'q' : 0.0125, 'clip' : (2.5, 97.5)},
    "meat_and_poultry_energy_ratio_target_day" : {'q':0.0, 'clip':(7.0, 93.0)},
    # ---------------- DENSITY INDICES ----------------
    "fiber_density_energy_target_day" : {'q' : 0.0, 'clip' : (5, 95)},
}
