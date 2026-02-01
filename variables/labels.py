labels_dict = {
    'total_sleep_time_minutes_target_day': 'Total Sleep Time',
    'percent_of_deep_sleep_time_target_day' : 'Deep Sleep %',
    'percent_of_rem_sleep_time_target_day' : 'REM Sleep %',
    'percent_of_light_sleep_time_target_day' : 'Light Sleep %',
    'sleep_efficiency_target_day' : 'Sleep Efficiency',
    'sleep_latency_minutes_target_day' : 'Sleep Onset Latency', 
    'heart_rate_mean_during_sleep_target_day': 'Mean Heart Rate',
    'total_wake_time_after_sleep_onset_minutes_target_day' : 'Wake After Sleep Onset',
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
    'night_calories_pct' : 'High-Calorie\nEvening\nMeal',
    
    'fat_pct' : 'Higher Fat Consumption (% of Total Energy Intake)',
    'carb_pct': 'Higher Carbs Consumption (% of Total Energy Intake)',
    'prot_pct': 'Higher Protein Consumption (% of Total Energy Intake)',
    
    'whole_food_categories_ratio' : 'Higher Whole Foods Consumption',
    'fish_meat_eggs_categories_ratio' : 'Animal\nFoods',
    'whole_dairy_categories_ratio' : 'Higher Dairy Consumption',
    'plant_based_whole_foods_ratio' : 'Higher\nPlant\nConsumption',
    #'processed_categories_ratio' : 'Processed\nFood',
    'fiber_g' : 'Fiber',
    
    'animal_based_whole_foods_ratio' : 'Higher Animal-Based Foods Consumption',
    'meat_and_poultry_energy_ratio' : 'Higher Meat and Poultry Consumption',
    'furits_and_veggies_energy_ratio' : 'Higher Fruits and Veggies Consumption',
    
    'processed_categories_ratio' : 'Higher Processed Food Consumption',
    
    'unique_plant_based_foods_count' : 'Higher\nPlant\nDiversity',
    'fiber_density_energy' : 'Fiber\nDensity',
    
    'caffeine_late_mg' : 'Late\nCaffeine',
    
    'magnesium_mg' : 'Higher Magnesium Consumption',
    'vitamin_b12_ug' : 'Higher B12 Consumption',
    'calcium_mg' : 'Higher Calcium Consumption',
    'zinc_mg' : 'Higher Zinc Consumption',
    'sugars_g': 'Higher Total Sugars Consumption',
    'vitamin_e': 'Higher Vitamin E Consumption',
    'vitamin_d_ug' : 'Higher Vitamin D Consumption',
    'folate' : 'Higher Folic Acid Consumption',
    'sat_fat_g': 'Higher Saturated Fat Consumption',
    'omega3_total_g' : 'Higher Omega-3 Consumption',
    
    'vitamin_b6_mg' : 'Higher B6 Consumption',
    'vitamin_c' : 'Higher Vitamin C Consumption',
}

rdis = {
    'fiber_g' : 30,
}

