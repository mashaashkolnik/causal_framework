labels_dict = {
    'composite_sleep_quality_target_day' : 'Composite Sleep Quality',
    'sleep_quality_continuity_target_day' : 'Sleep Quality Continuity',
    'sleep_quality_architecture_target_day': 'Sleep Quality Architecture',
    'sleep_latency_minutes_target_day' : 'Sleep Onset Latency', 
    'sleep_efficiency_target_day' : 'Sleep Efficiency',
    'total_wake_time_after_sleep_onset_minutes_target_day' : 'Wake After Sleep Onset',
    'total_arousal_sleep_time_target_day' : 'Arousals Sleep Time',
    'number_of_wakes_target_day' : 'Number of wakes',
    'percent_of_deep_sleep_time_target_day' : 'Deep Sleep %',
    'percent_of_rem_sleep_time_target_day' : 'REM Sleep %',
    'percent_of_light_sleep_time_target_day' : 'Light Sleep %',
    'total_deep_sleep_time_minutes_target_day': 'Deep Sleep Time',
    'total_rem_sleep_time_minutes_target_day': 'REM Sleep Time',
    'total_sleep_time_minutes_target_day': 'Total Sleep Time',
    'heart_rate_mean_during_sleep_target_day': 'Mean Heart Rate',
}

annotation_dict = {
    'sleep_latency_minutes_target_day' : 'min', 
    'sleep_efficiency_target_day' : 'pp',
    'total_wake_time_after_sleep_onset_minutes_target_day' : 'min',
    'number_of_wakes_target_day' : '',
    'percent_of_deep_sleep_time_target_day' : 'pp',
    'percent_of_rem_sleep_time_target_day' : 'pp',
    'percent_of_light_sleep_time_target_day' : 'pp',
    'total_sleep_time_minutes_target_day': 'min',
    'heart_rate_mean_during_sleep_target_day': 'bpm',
}

# Treated vs Control boxplots
# Figure 2 
diet_short_names_mapping = {
    'hours_to_sleep' : 'Last-meal–to-bedtime interval (h)',
    'eating_window_h' : 'Eating window duration (h)',
    'night_calories_pct' : 'Dinner energy (%)',
    
    'fat_pct' : 'Fat (% of daily energy)',
    'carb_pct': 'Carbs (% of daily energy)',
    'prot_pct': 'Protein (% of daily energy)',
    
    'whole_food_categories_ratio' : 'Whole foods (% of daily energy)',
    'animal_based_whole_foods_ratio' : 'Animal-based foods (% of daily energy)',
    'meat_and_poultry_energy_ratio' : 'Meat and poultry (% of daily energy)',
    'furits_and_veggies_energy_ratio' : 'Fruits and vegetables (% of daily energy)',
    'whole_dairy_categories_ratio' : 'Dairy (% of daily energy)',
    'plant_based_whole_foods_ratio' : 'Plant-based foods (% of daily energy)',
    'processed_categories_ratio' : 'Processed foods (% of daily energy)',
    'unique_plant_based_foods_count' : 'Plant-based food diversity (count)',
    
    #'caffeine_late_mg' : 'Late\nCaffeine',
        
    'fiber_g' : 'Fiber',
    'fiber_density_energy' : 'Dietary fiber density (g/1,000 kcal)',
    
    'magnesium_mg' : 'Magnesium (mg)',
    'vitamin_b12_ug' : 'Vitamin b12 (ug)',
    'calcium_mg' : 'Calcium (mg)',
    'zinc_mg' : 'Zinc (mg)',
    'sugars_g': 'Sugars (g)',
    'vitamin_e': 'Vitamin E (mg)',
    'vitamin_d_ug' : 'Vitamin D (mcg)',
    'folate' : 'Folic Acid (mcg)',
    'sat_fat_g': 'Saturated Fat (g)',
    'omega3_total_g' : 'Omega-3 (g)',
    'vitamin_b6_mg' : 'Vitamin b6 (mg)',
    'vitamin_c' : 'Vitamin C (mg)',
}

# Table
diet_definitions = {
    'hours_to_sleep' : 'Time Between Last Meal and Bedtime, hours',
    'eating_window_h' : 'Time Between First and Last Meals, hours',
    'night_calories_pct' : '% Calories\nLate',
    
    'fat_pct' : 'Fat, % of daily energy intake',
    'carb_pct': 'Carbs, % of daily energy intake',
    'prot_pct': 'Protein, % of daily energy intake',
    
    'whole_food_categories_ratio' : 'Whole foods, % of daily energy intake',
    'animal_based_whole_foods_ratio' : 'Animal-based foods, % of daily energy intake',
    'meat_and_poultry_energy_ratio' : 'Meat and Poultry, % of daily energy intake',
    'furits_and_veggies_energy_ratio' : 'Fruits and Veggies, % of daily energy intake',
    'whole_dairy_categories_ratio' : 'Dairy, % of daily energy intake',
    'plant_based_whole_foods_ratio' : 'Plant-based foods, % of daily energy intake',
    'processed_categories_ratio' : 'Processed food, % of daily energy intake',
    
    'unique_plant_based_foods_count' : 'Unique plant sources',
    'fiber_density_energy' : 'Fiber\nDensity',
    
    'caffeine_late_mg' : 'Late caffeine, mg',
    
    'fiber_g' : 'Fiber, g',
    'fiber_density_energy' : 'Fiber Density',
    'magnesium_mg' : 'Magnesium, mg',
    'vitamin_b12_ug' : 'Vitamin b12, ug',
    'calcium_mg' : 'Calcium, mg',
    'zinc_mg' : 'Zinc, mg',
    'sugars_g': 'Sugars, g',
    'vitamin_e': 'Vitamin E',
    'vitamin_d_ug' : 'Vitamin D',
    'folate' : 'Folic Acid',
    'sat_fat_g': 'Saturated Fat, g',
    'omega3_total_g' : 'Omega-3, g',
    'vitamin_b6_mg' : 'Vitamin b6, mg',
    'vitamin_c' : 'Vitamin C',
}

diet_full_names_mapping = {
    'hours_to_sleep' : 'Earlier\nDinner',
    'eating_window_h' : 'Longer\nEating\nWindow',
    'night_calories_pct' : 'Heavy\nEvening\nMeal',
    
    'fat_pct' : 'Higher Fat Consumption (% of Total Energy Intake)',
    'carb_pct': 'Higher Carbs Consumption (% of Total Energy Intake)',
    'prot_pct': 'Higher Protein Consumption (% of Total Energy Intake)',
    
    'whole_food_categories_ratio' : 'Higher whole foods Consumption (% of Total Energy Intake)',
    'fish_meat_eggs_categories_ratio' : 'Animal\nFoods',
    'whole_dairy_categories_ratio' : 'Higher Dairy Consumption (% of Total Energy Intake)',
    'plant_based_whole_foods_ratio' : 'Whole\nPlant',
    'processed_categories_ratio' : 'Processed\nFood',
    'fiber_g' : 'Fiber',
    
    'unique_plant_based_foods_count' : 'High\nPlant\nDiversity',
    'fiber_density_energy' : 'Fiber\nDensity',
    
    'caffeine_late_mg' : 'Late\nCaffeine',
    
    'magnesium_mg' : 'Mg',
    'vitamin_b12_ug' : 'B12',
    'calcium_mg' : 'Higher Calcium Consumption (mg)',
    'zinc_mg' : 'Higher Zinc Consumption (mg)',
    'sugars_g': 'Higher Total Sugars Consumption (g)',
    'vitamin_e': 'Higher Vitamin E Consumption (mg)',
    'vitamin_d_ug' : 'Higher Vitamin D Consumption (ug)',
    'folate' : 'Higher Folate Consumption',
    'sat_fat_g': 'Higher Saturated Fat Consumption (g)',
    'omega3_total_g' : 'Higher Omega-3 Consumption (g)',
    
    'vitamin_b6_mg' : 'B6',
    'vitamin_c' : 'Vit C',
}

rdis = {
    'fiber_g' : 30,
}

