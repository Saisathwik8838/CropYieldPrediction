"""
Make single crop yield predictions using the trained model
"""
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def predict_single():
    print("🔮 Crop Yield Prediction - Interactive Mode\n")
    
    # Define paths
    model_file = Path('crop_yield_model.pkl')
    
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
        target_column = model_data['target_column']
        label_encoders = model_data.get('label_encoders', {})
        numeric_features = model_data['numeric_features']
        categorical_features = model_data['categorical_features']
        training_info = model_data['training_info']
        
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Target variable: '{target_column}'")
        print(f"   ✅ Model expects {len(feature_columns)} features")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Step 2: Display model information
    print("\n📊 Model Information:")
    print("="*60)
    print(f"   Training samples: {training_info['n_samples']}")
    print(f"   Number of features: {training_info['n_features']}")
    print(f"   Target range: {training_info['target_min']:.2f} to {training_info['target_max']:.2f}")
    print(f"   Target mean: {training_info['target_mean']:.2f}")
    
    if 'metrics' in model_data:
        metrics = model_data['metrics']
        print(f"\n   Model Performance:")
        print(f"   - Test R² Score: {metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"   - Test MAE: {metrics['test_mae']:.2f}")
    
    print("="*60)
    
    # Step 3: Display important features
    if 'feature_importance' in model_data:
        print("\n🎯 Top 5 Most Important Features:")
        for i, feat_info in enumerate(model_data['feature_importance'][:5], 1):
            print(f"   {i}. {feat_info['feature']} (importance: {feat_info['importance']:.4f})")
    
    # Step 4: Get user input for prediction
    print("\n📝 Enter values for prediction:")
    print("-"*60)
    print("   (Press Enter to use default value for optional fields)\n")
    
    input_data = {}
    display_data = {}  # For showing decoded values
    
    # Collect input for each feature
    for feature in feature_columns:
        while True:
            try:
                # Check if feature is categorical or numeric
                if feature in categorical_features:
                    # Categorical feature
                    print(f"\n{feature} (categorical)")
                    
                    # Get available categories from label encoder
                    if feature in label_encoders:
                        categories = list(label_encoders[feature].classes_)
                        
                        # Show first 10 options
                        print(f"   Available options: {', '.join(map(str, categories[:10]))}")
                        if len(categories) > 10:
                            print(f"   ... and {len(categories) - 10} more options")
                    
                    value = input(f"   Enter {feature}: ").strip()
                    
                    if value == '':
                        # Use default (first category or 'unknown')
                        if feature in label_encoders and len(label_encoders[feature].classes_) > 0:
                            default_val = label_encoders[feature].classes_[0]
                            input_data[feature] = 0
                            display_data[feature] = default_val
                            print(f"   ℹ️  Using default: {default_val}")
                        else:
                            input_data[feature] = 0
                            display_data[feature] = 'unknown'
                    else:
                        # Encode the value
                        if feature in label_encoders:
                            try:
                                encoded_value = label_encoders[feature].transform([value])[0]
                                input_data[feature] = encoded_value
                                display_data[feature] = value
                            except ValueError:
                                print(f"   ⚠️  '{value}' not in training data, using default")
                                input_data[feature] = 0
                                display_data[feature] = f"{value} (unknown)"
                        else:
                            input_data[feature] = 0
                            display_data[feature] = value
                    
                elif feature in numeric_features:
                    # Numeric feature
                    value = input(f"   Enter {feature} (numeric): ").strip()
                    
                    if value == '':
                        # Use default value (0)
                        input_data[feature] = 0
                        display_data[feature] = 0
                        print(f"   ℹ️  Using default value: 0")
                    else:
                        numeric_val = float(value)
                        input_data[feature] = numeric_val
                        display_data[feature] = numeric_val
                else:
                    # Unknown type, try to infer
                    value = input(f"   Enter {feature}: ").strip()
                    
                    if value == '':
                        input_data[feature] = 0
                        display_data[feature] = 0
                    else:
                        try:
                            numeric_val = float(value)
                            input_data[feature] = numeric_val
                            display_data[feature] = numeric_val
                        except ValueError:
                            input_data[feature] = value
                            display_data[feature] = value
                
                break
                
            except ValueError:
                print(f"   ❌ Invalid input. Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\n   ⚠️  Prediction cancelled by user")
                return
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print(f"   Please try again.")
    
    # Step 5: Create dataframe for prediction
    print("\n🔧 Preparing data for prediction...")
    
    try:
        # Create a dataframe with a single row
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in the correct order
        input_df = input_df[feature_columns]
        
        print("   ✅ Input data prepared")
        
    except Exception as e:
        print(f"   ❌ Error preparing input data: {e}")
        return
    
    # Step 6: Make prediction
    print("\n🎯 Making prediction...")
    
    try:
        prediction = model.predict(input_df)[0]
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\n   Predicted {target_column}: {prediction:,.2f}")
        
        # Show prediction confidence range if available
        if hasattr(model, 'estimators_'):
            # For ensemble models, calculate prediction std
            predictions_all = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
            pred_std = predictions_all.std()
            
            print(f"   Confidence range: {prediction - pred_std:,.2f} to {prediction + pred_std:,.2f}")
            print(f"   (±{pred_std:,.2f})")
        
        print("\n" + "="*60)
        
        # Display input summary
        print("\n📋 Input Summary:")
        print("-"*60)
        for feature in feature_columns:
            print(f"   {feature}: {display_data[feature]}")
        
        print("="*60)
        
        # Save prediction to file (optional)
        save_result = input("\n💾 Save this prediction to file? (yes/no): ").strip().lower()
        
        if save_result in ['yes', 'y']:
            try:
                result_df = pd.DataFrame([{
                    **display_data,
                    f'predicted_{target_column}': prediction,
                    'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                
                output_file = Path('data/single_predictions.csv')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Append to existing file or create new
                if output_file.exists():
                    existing_df = pd.read_csv(output_file)
                    result_df = pd.concat([existing_df, result_df], ignore_index=True)
                
                result_df.to_csv(output_file, index=False)
                print(f"   ✅ Prediction saved to {output_file}")
            except Exception as e:
                print(f"   ⚠️  Could not save prediction: {e}")
        
        # Offer to make another prediction
        print("\n")
        another = input("Would you like to make another prediction? (yes/no): ").strip().lower()
        
        if another in ['yes', 'y']:
            print("\n" + "="*60 + "\n")
            predict_single()
        else:
            print("\n✅ Thank you for using the Crop Yield Predictor!")
        
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return

def predict_from_dict(input_dict, model_path='crop_yield_model.pkl'):
    """
    Function to make a prediction from a dictionary of input values
    
    Args:
        input_dict (dict): Dictionary with feature names as keys and values
        model_path (str): Path to the saved model file
    
    Returns:
        float: Predicted value
    
    Example:
        input_data = {
            'area': 1500,
            'year': 2024,
            'crop': 'wheat',
            'state': 'Punjab'
        }
        prediction = predict_from_dict(input_data)
        print(f"Predicted yield: {prediction:.2f}")
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    try:
        # Load model
        model_data = joblib.load(model_file)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoders = model_data.get('label_encoders', {})
        categorical_features = model_data['categorical_features']
        numeric_features = model_data['numeric_features']
        
        # Standardize input keys to lowercase
        input_dict_lower = {k.lower().strip(): v for k, v in input_dict.items()}
        
        # Create input dataframe
        input_data = {}
        
        for feature in feature_columns:
            if feature in input_dict_lower:
                value = input_dict_lower[feature]
                
                # Handle categorical encoding
                if feature in categorical_features and feature in label_encoders:
                    try:
                        input_data[feature] = label_encoders[feature].transform([str(value)])[0]
                    except ValueError:
                        # Unknown category, use default
                        input_data[feature] = 0
                else:
                    input_data[feature] = value
            else:
                # Use default value for missing features
                if feature in categorical_features:
                    input_data[feature] = 0
                else:
                    input_data[feature] = 0
        
        # Create dataframe and predict
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]
        
        prediction = model.predict(input_df)[0]
        
        return prediction
        
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")

if __name__ == "__main__":
    predict_single()
    
    # Example of using the dictionary prediction function:
    # print("\n" + "="*60)
    # print("Example: Programmatic Prediction")
    # print("="*60)
    # sample_input = {
    #     'area': 1500,
    #     'year': 2024,
    #     'crop': 'wheat',
    #     'state': 'Punjab'
    # }
    # try:
    #     prediction = predict_from_dict(sample_input)
    #     print(f"Input: {sample_input}")
    #     print(f"Predicted yield: {prediction:.2f}")
    # except Exception as e:
    #     print(f"Error: {e}")