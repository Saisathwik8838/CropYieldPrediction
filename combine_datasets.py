"""
Combine all crop data with weather data into a single dataset
"""
import pandas as pd
import os
from pathlib import Path

def combine_datasets():
    print("🔄 Starting dataset combination process...")
    
    # Define paths
    crops_dir = Path('data/Crops')
    temp_file = Path('data/temperature.csv')
    rainfall_file = Path('data/rainfall.csv')
    output_file = Path('data/final_dataset.csv')
    
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)
    
    # Step 1: Read all crop CSV files
    print("📂 Reading crop data files...")
    crop_dataframes = []
    
    if not crops_dir.exists():
        print(f"⚠️  Warning: {crops_dir} directory not found. Creating it...")
        crops_dir.mkdir(parents=True, exist_ok=True)
        print("⚠️  Please add crop CSV files to data/Crops/ directory")
        return
    
    crop_files = list(crops_dir.glob('*.csv'))
    
    if not crop_files:
        print("⚠️  Warning: No CSV files found in data/Crops/ directory")
        print("⚠️  Please add crop CSV files to continue")
        return
    
    for crop_file in crop_files:
        try:
            df = pd.read_csv(crop_file)
            crop_name = crop_file.stem
            df['crop'] = crop_name
            crop_dataframes.append(df)
            print(f"   ✅ Loaded {crop_file.name} - {len(df)} rows")
        except Exception as e:
            print(f"   ❌ Error loading {crop_file.name}: {e}")
    
    if not crop_dataframes:
        print("❌ No crop data loaded successfully")
        return
    
    # Combine all crop data
    print("\n🔗 Combining crop data...")
    combined_crops = pd.concat(crop_dataframes, ignore_index=True)
    print(f"   ✅ Combined dataset: {len(combined_crops)} rows")
    
    # Step 2: Load weather data
    final_df = combined_crops.copy()
    
    # Standardize column names (lowercase, strip spaces)
    final_df.columns = final_df.columns.str.lower().str.strip()
    
    # Try to merge temperature data
    if temp_file.exists():
        try:
            print("\n🌡️  Loading temperature data...")
            temp_df = pd.read_csv(temp_file)
            temp_df.columns = temp_df.columns.str.lower().str.strip()
            
            # Find common columns for merging
            common_cols = list(set(final_df.columns) & set(temp_df.columns) - {'crop'})
            
            if common_cols:
                print(f"   Merging on columns: {common_cols}")
                final_df = final_df.merge(temp_df, on=common_cols, how='left', suffixes=('', '_temp'))
                print(f"   ✅ Temperature data merged")
            else:
                print("   ⚠️  No common columns found for temperature merge")
        except Exception as e:
            print(f"   ❌ Error loading temperature data: {e}")
    else:
        print(f"\n⚠️  Temperature file not found: {temp_file}")
    
    # Try to merge rainfall data
    if rainfall_file.exists():
        try:
            print("\n🌧️  Loading rainfall data...")
            rain_df = pd.read_csv(rainfall_file)
            rain_df.columns = rain_df.columns.str.lower().str.strip()
            
            # Find common columns for merging
            common_cols = list(set(final_df.columns) & set(rain_df.columns) - {'crop'})
            
            if common_cols:
                print(f"   Merging on columns: {common_cols}")
                final_df = final_df.merge(rain_df, on=common_cols, how='left', suffixes=('', '_rain'))
                print(f"   ✅ Rainfall data merged")
            else:
                print("   ⚠️  No common columns found for rainfall merge")
        except Exception as e:
            print(f"   ❌ Error loading rainfall data: {e}")
    else:
        print(f"\n⚠️  Rainfall file not found: {rainfall_file}")
    
    # Step 3: Clean the data
    print("\n🧹 Cleaning data...")
    
    # Fill missing numeric values with mean
    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if final_df[col].isnull().any():
            mean_val = final_df[col].mean()
            final_df[col].fillna(mean_val, inplace=True)
            print(f"   ✅ Filled missing values in '{col}' with mean: {mean_val:.2f}")
    
    # Fill missing categorical values with mode or 'Unknown'
    categorical_cols = final_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if final_df[col].isnull().any():
            mode_val = final_df[col].mode()
            if len(mode_val) > 0:
                final_df[col].fillna(mode_val[0], inplace=True)
                print(f"   ✅ Filled missing values in '{col}' with mode: {mode_val[0]}")
            else:
                final_df[col].fillna('Unknown', inplace=True)
                print(f"   ✅ Filled missing values in '{col}' with 'Unknown'")
    
    # Step 4: Save the final dataset
    print(f"\n💾 Saving combined dataset to {output_file}...")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined dataset saved successfully!")
    print(f"   📊 Total rows: {len(final_df)}")
    print(f"   📊 Total columns: {len(final_df.columns)}")
    print(f"   📊 Columns: {list(final_df.columns)}")
    print(f"\n{'='*60}")
    print("Dataset summary:")
    print(final_df.info())

if __name__ == "__main__":
    combine_datasets()