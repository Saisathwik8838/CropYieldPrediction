"""
Generate sample data for testing the crop yield prediction project
Run this if you don't have your own data yet
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data():
    print("🌾 Generating sample data for testing...\n")
    
    # Create directories
    Path('data/Crops').mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define parameters
    states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka']
    years = list(range(2015, 2025))
    
    # Generate crop data
    crops = {
        'wheat': {'base_yield': 3000, 'variation': 500},
        'rice': {'base_yield': 2500, 'variation': 400},
        'corn': {'base_yield': 3500, 'variation': 600},
        'cotton': {'base_yield': 1800, 'variation': 300},
        'sugarcane': {'base_yield': 7000, 'variation': 1000}
    }
    
    print("📊 Generating crop datasets...")
    
    for crop_name, params in crops.items():
        data = []
        
        for state in states:
            for year in years:
                # Generate realistic data with some patterns
                area = np.random.randint(500, 3000)
                
                # Yield influenced by area, year trend, and random variation
                year_factor = (year - 2015) * 50  # Slight improvement over years
                area_factor = area * 0.5  # Larger areas tend to have better yields
                random_factor = np.random.normal(0, params['variation'])
                
                value = params['base_yield'] + year_factor + area_factor + random_factor
                value = max(0, value)  # Ensure non-negative
                
                data.append({
                    'State': state,
                    'Year': year,
                    'Area': area,
                    'Value': round(value, 2)
                })
        
        df = pd.DataFrame(data)
        output_file = Path(f'data/Crops/{crop_name}.csv')
        df.to_csv(output_file, index=False)
        print(f"   ✅ {crop_name}.csv created - {len(df)} rows")
    
    # Generate temperature data
    print("\n🌡️  Generating temperature data...")
    temp_data = []
    
    for state in states:
        for year in years:
            # Temperature varies by state and year
            base_temp = {
                'Punjab': 25, 'Haryana': 26, 'Uttar Pradesh': 27,
                'Maharashtra': 28, 'Karnataka': 26
            }[state]
            
            # Add seasonal variation and climate change trend
            temp = base_temp + (year - 2015) * 0.1 + np.random.normal(0, 2)
            
            temp_data.append({
                'State': state,
                'Year': year,
                'Avg_Temperature': round(temp, 2),
                'Min_Temperature': round(temp - 5, 2),
                'Max_Temperature': round(temp + 8, 2)
            })
    
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv('data/temperature.csv', index=False)
    print(f"   ✅ temperature.csv created - {len(temp_df)} rows")
    
    # Generate rainfall data
    print("\n🌧️  Generating rainfall data...")
    rainfall_data = []
    
    for state in states:
        for year in years:
            # Rainfall varies by state
            base_rainfall = {
                'Punjab': 600, 'Haryana': 550, 'Uttar Pradesh': 1000,
                'Maharashtra': 1200, 'Karnataka': 900
            }[state]
            
            # Add yearly variation
            rainfall = base_rainfall + np.random.normal(0, 150)
            rainfall = max(0, rainfall)
            
            rainfall_data.append({
                'State': state,
                'Year': year,
                'Annual_Rainfall': round(rainfall, 2),
                'Monsoon_Rainfall': round(rainfall * 0.7, 2),
                'Winter_Rainfall': round(rainfall * 0.3, 2)
            })
    
    rainfall_df = pd.DataFrame(rainfall_data)
    rainfall_df.to_csv('data/rainfall.csv', index=False)
    print(f"   ✅ rainfall.csv created - {len(rainfall_df)} rows")
    
    # Generate sample batch input
    print("\n📋 Generating sample batch input...")
    batch_data = {
        'Area': [1000, 1500, 2000, 1200, 1800, 2500],
        'Year': [2024, 2024, 2024, 2025, 2025, 2025],
        'Crop': ['wheat', 'rice', 'corn', 'wheat', 'cotton', 'sugarcane'],
        'State': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Punjab']
    }
    
    batch_df = pd.DataFrame(batch_data)
    batch_df.to_csv('data/batch_input.csv', index=False)
    print(f"   ✅ batch_input.csv created - {len(batch_df)} rows")
    
    # Display summary
    print("\n" + "="*60)
    print("✅ SAMPLE DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"   Created files:")
    print(f"   - data/Crops/wheat.csv")
    print(f"   - data/Crops/rice.csv")
    print(f"   - data/Crops/corn.csv")
    print(f"   - data/Crops/cotton.csv")
    print(f"   - data/Crops/sugarcane.csv")
    print(f"   - data/temperature.csv")
    print(f"   - data/rainfall.csv")
    print(f"   - data/batch_input.csv")
    print(f"\n   You can now run:")
    print(f"   1. python combine_datasets.py")
    print(f"   2. python model_train.py")
    print(f"   3. python model_predict.py")
    print(f"   4. python model_batch_predict.py")
    print("="*60)
    
    # Show sample of generated data
    print("\n📊 Sample of wheat.csv:")
    wheat_df = pd.read_csv('data/Crops/wheat.csv')
    print(wheat_df.head())
    
    print("\n📊 Sample of temperature.csv:")
    print(temp_df.head())
    
    print("\n📊 Sample of rainfall.csv:")
    print(rainfall_df.head())

if __name__ == "__main__":
    generate_sample_data()