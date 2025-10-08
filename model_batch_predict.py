"""
Predict crop yield for multiple rows from a CSV file
"""
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def batch_predict():
    print("🔮 Crop Yield Prediction - Batch Processing\n")
    
    # Define paths
    model_file = Path('crop_yield_model.pkl')
    batch_input_file = Path('data/batch_input.csv')
    output_file = Path('data/batch_predictions.csv')
    
    # Step 1: Load the model
    print("📂 Loading trained model...")
    
    if not model_file.exists():
        print(f"❌ Error: Model file not found at {model_file}")
        print("   Please run model_train.py first to train and save the model")
        return
    
    try:
        model_data = joblib.load(model_file)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        numeric_features = model_data['numeric_features']
        categorical_features = model_data['categorical_features']
        target_column = model_data['target_column']
        label_encoders = model_data.get('label_encoders', {})
        
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Target variable: '{target_column}'")
        print(f"   ✅ Expected features ({len(feature_columns)}): {feature_columns[:5]}...")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Step 2: Load batch input CSV
    print(f"\n📂 Loading batch input from {batch_input_file}...")
    
    if not batch_input_file.exists():
        print(f"   ⚠️  File not found: {batch_input_file}")
        print(f"   ℹ️  Creating sample batch_input.csv file...")
        
        # Create a sample batch input file
        sample_data = {
            'area': [1000, 1500, 2000, 1200, 1800],
            'year': [2024, 2024, 2023, 2024, 2023],
            'crop': ['wheat', 'rice', 'wheat', 'corn', 'rice']
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Create data directory if needed
        batch_input_file.parent.mkdir(parents=True, exist_ok=True)
        
        sample_df.to_csv(batch_input_file, index=False)
        print(f"   ✅ Sample file created with {len(sample_df)} rows")
        print(f"   ℹ️  Please update {batch_input_file} with your actual data and run again")
        return
    
    try:
        batch_df = pd.read_csv(batch_input_file)
        
        if batch_df.empty:
            print(f"   ❌ Error: {batch_input_file} is empty")
            return
        
        print(f"   ✅ Loaded {len(batch_df)} rows for prediction")
        print(f"   ✅ Input columns: {list(batch_df.columns)}")
        
    except Exception as e:
        print(f"   ❌ Error loading batch input: {e}")
        return
    
    # Step 3: Prepare data for prediction
    print("\n🔧 Preparing batch data for prediction...")
    
    # Store original data for output
    original_batch_df = batch_df.copy()
    
    # Standardize column names
    batch_df.columns = batch_df.columns.str.lower().str.strip()
    
    # Check for required key columns
    key_columns = ['area', 'year', 'crop']
    available_keys = [col for col in key_columns if col in batch_df.columns]
    missing_key_cols = [col for col in key_columns if col not in batch_df.columns]
    
    if missing_key_cols:
        print(f"   ⚠️  Warning: Missing key columns: {missing_key_cols}")
        print(f"   ℹ️  Available columns: {list(batch_df.columns)}")
    
    # Identify missing features
    missing_features = set(feature_columns) - set(batch_df.columns)
    
    if missing_features:
        print(f"\n   ⚠️  Missing {len(missing_features)} features in input CSV:")
        print(f"   {list(missing_features)[:10]}{'...' if len(missing_features) > 10 else ''}")
        print(f"   ℹ️  Adding missing features with default values...")
        
        # Add missing features with appropriate defaults
        for feature in missing_features:
            # Determine if feature is numeric or categorical based on training data
            if feature in numeric_features:
                # Use 0 for numeric features
                default_value = 0
                batch_df[feature] = default_value
            elif feature in categorical_features:
                # Use 'unknown' for categorical features
                batch_df[feature] = 'unknown'
            else:
                # If we can't determine, guess based on name
                if any(keyword in feature.lower() for keyword in ['temp', 'rain', 'area', 'year', 'production', 'value']):
                    batch_df[feature] = 0
                else:
                    batch_df[feature] = 'unknown'
        
        print(f"   ✅ Missing features added with defaults")
    
    # Handle categorical encoding
    print("\n🔤 Encoding categorical variables...")
    for col in categorical_features:
        if col in batch_df.columns and col in label_encoders:
            try:
                # Convert to string and encode
                batch_df[col] = batch_df[col].astype(str)
                
                # Encode values, using 0 for unknown categories
                encoded_values = []
                for val in batch_df[col]:
                    try:
                        encoded_val = label_encoders[col].transform([val])[0]
                        encoded_values.append(encoded_val)
                    except ValueError:
                        # Unknown category, use 0
                        encoded_values.append(0)
                
                batch_df[col] = encoded_values
                print(f"   ✅ Encoded '{col}'")
            except Exception as e:
                print(f"   ⚠️  Warning encoding '{col}': {e}")
                batch_df[col] = 0
    
    # Ensure columns are in the correct order
    try:
        batch_df = batch_df[feature_columns]
        print(f"   ✅ Columns reordered to match training data")
    except KeyError as e:
        print(f"   ❌ Error: Could not reorder columns - {e}")
        return
    
    # Step 4: Check for invalid rows
    print("\n🔍 Checking data quality...")
    
    # Check for rows with all missing values
    valid_rows = []
    invalid_rows = []
    
    for idx, row in batch_df.iterrows():
        # Check if row has critical missing values
        if pd.isna(row).all():
            invalid_rows.append(idx)
        else:
            valid_rows.append(idx)
    
    if invalid_rows:
        print(f"   ⚠️  Found {len(invalid_rows)} invalid rows (will be skipped)")
        batch_df = batch_df.loc[valid_rows]
        original_batch_df = original_batch_df.loc[valid_rows]
    
    if batch_df.empty:
        print(f"   ❌ Error: No valid rows remaining after filtering")
        return
    
    print(f"   ✅ {len(batch_df)} valid rows ready for prediction")
    
    # Step 5: Make predictions
    print("\n🎯 Making predictions...")
    print("   (This may take a moment for large datasets...)")
    
    try:
        predictions = model.predict(batch_df)
        print(f"   ✅ Predictions completed for {len(predictions)} rows")
        
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        return
    
    # Step 6: Prepare output
    print("\n📊 Preparing results...")
    
    # Add predictions to original dataframe
    results_df = original_batch_df.copy()
    results_df[f'predicted_{target_column}'] = predictions
    
    # Add prediction confidence/info if available
    results_df['prediction_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Display summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Mean predicted {target_column}: {predictions.mean():,.2f}")
    print(f"   Min predicted {target_column}: {predictions.min():,.2f}")
    print(f"   Max predicted {target_column}: {predictions.max():,.2f}")
    print(f"   Std deviation: {predictions.std():,.2f}")
    print("="*60)
    
    # Step 7: Save results
    print(f"\n💾 Saving predictions to {output_file}...")
    
    try:
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_file, index=False)
        print(f"   ✅ Results saved successfully!")
        
        # Display first few predictions
        print("\n📋 Preview of results (first 5 rows):")
        print("="*60)
        
        display_cols = list(original_batch_df.columns) + [f'predicted_{target_column}']
        print(results_df[display_cols].head().to_string(index=False))
        
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
        return
    
    print("\n" + "="*60)
    print("✅ BATCH PREDICTION COMPLETE!")
    print("="*60)
    print(f"   Input file: {batch_input_file}")
    print(f"   Output file: {output_file}")
    print(f"   Predictions: {len(predictions)}")
    print("="*60)

def predict_from_dataframe(df, model_path='crop_yield_model.pkl'):
    """
    Function to make predictions from a pandas DataFrame
    
    Args:
        df (pd.DataFrame): Input dataframe with features
        model_path (str): Path to the saved model file
    
    Returns:
        np.ndarray: Array of predictions
    
    Example:
        batch_data = pd.DataFrame({
            'area': [1000, 1500, 2000],
            'year': [2024, 2024, 2024],
            'crop': ['wheat', 'rice', 'corn']
        })
        predictions = predict_from_dataframe(batch_data)
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    try:
        # Load model
        model_data = joblib.load(model_file)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        numeric_features = model_data['numeric_features']
        categorical_features = model_data['categorical_features']
        label_encoders = model_data.get('label_encoders', {})
        
        # Prepare dataframe
        input_df = df.copy()
        
        # Standardize column names
        input_df.columns = input_df.columns.str.lower().str.strip()
        
        # Add missing features
        missing_features = set(feature_columns) - set(input_df.columns)
        for feature in missing_features:
            if feature in numeric_features:
                input_df[feature] = 0
            elif feature in categorical_features:
                input_df[feature] = 'unknown'
            else:
                if any(keyword in feature.lower() for keyword in ['temp', 'rain', 'area', 'year', 'production', 'value']):
                    input_df[feature] = 0
                else:
                    input_df[feature] = 'unknown'
        
        # Handle categorical encoding
        for col in categorical_features:
            if col in input_df.columns and col in label_encoders:
                input_df[col] = input_df[col].astype(str)
                
                # Encode values, using 0 for unknown categories
                encoded_values = []
                for val in input_df[col]:
                    try:
                        encoded_val = label_encoders[col].transform([val])[0]
                        encoded_values.append(encoded_val)
                    except ValueError:
                        encoded_values.append(0)
                
                input_df[col] = encoded_values
        
        # Reorder columns
        input_df = input_df[feature_columns]
        
        # Make predictions
        predictions = model.predict(input_df)
        
        return predictions
        
    except Exception as e:
        raise Exception(f"Error during batch prediction: {e}")

if __name__ == "__main__":
    batch_predict()
    
    # Example of using the dataframe prediction function:
    # custom_batch = pd.DataFrame({
    #     'area': [1000, 1500, 2000],
    #     'year': [2024, 2024, 2024],
    #     'crop': ['wheat', 'rice', 'corn']
    # })
    # predictions = predict_from_dataframe(custom_batch)
    # print(f"\nCustom batch predictions: {predictions}")