"""
Train a machine learning model to predict crop yield
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("🚀 Starting Crop Yield Model Training\n")
    
    # Define paths
    dataset_file = Path('data/final_dataset.csv')
    model_file = Path('crop_yield_model.pkl')
    
    # Step 1: Load the dataset
    print("📂 Loading dataset...")
    
    if not dataset_file.exists():
        print(f"❌ Error: Dataset not found at {dataset_file}")
        print("   Please run combine_datasets.py first to create the dataset")
        return
    
    try:
        df = pd.read_csv(dataset_file)
        print(f"   ✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"   ✅ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return
    
    # Step 2: Identify target variable
    print("\n🎯 Identifying target variable...")
    
    # Common target column names
    target_candidates = ['value', 'yield', 'production', 'crop_yield', 'target']
    target_column = None
    
    for candidate in target_candidates:
        if candidate in df.columns:
            target_column = candidate
            break
    
    if target_column is None:
        # If no standard target found, use the last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_column = numeric_cols[-1]
            print(f"   ⚠️  No standard target column found, using '{target_column}'")
        else:
            print("   ❌ Error: No numeric columns found for prediction target")
            return
    
    print(f"   ✅ Target variable: '{target_column}'")
    
    # Step 3: Prepare features and target
    print("\n🔧 Preparing features and target...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Remove any rows with missing target values
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"   ✅ Valid samples: {len(X)}")
    print(f"   ✅ Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"   ✅ Target mean: {y.mean():.2f}")
    
    # Step 4: Handle categorical variables
    print("\n🔤 Encoding categorical variables...")
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"   Found {len(categorical_cols)} categorical features: {categorical_cols}")
    print(f"   Found {len(numeric_cols)} numeric features: {numeric_cols}")
    
    # Store label encoders for later use
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)  # Convert to string to handle any type
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   ✅ Encoded '{col}': {len(le.classes_)} unique values")
    
    # Step 5: Handle missing values
    print("\n🧹 Handling missing values...")
    
    missing_counts = X.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"   Found missing values in {len(cols_with_missing)} columns:")
        for col, count in cols_with_missing.items():
            print(f"      - {col}: {count} missing ({count/len(X)*100:.1f}%)")
        
        # Fill missing values
        for col in X.columns:
            if X[col].isnull().any():
                if col in numeric_cols:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
        
        print(f"   ✅ Missing values filled")
    else:
        print(f"   ✅ No missing values found")
    
    # Step 6: Split the data
    print("\n✂️  Splitting data into train and test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   ✅ Training set: {len(X_train)} samples")
    print(f"   ✅ Test set: {len(X_test)} samples")
    
    # Step 7: Train the model
    print("\n🤖 Training Random Forest model...")
    print("   (This may take a few moments...)")
    
    # Initialize model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Train the model
    model.fit(X_train, y_train)
    print(f"   ✅ Model training completed")
    
    # Step 8: Evaluate the model
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"{'Metric':<20} {'Training':<20} {'Testing':<20}")
    print("-"*60)
    print(f"{'RMSE':<20} {train_rmse:<20.2f} {test_rmse:<20.2f}")
    print(f"{'MAE':<20} {train_mae:<20.2f} {test_mae:<20.2f}")
    print(f"{'R² Score':<20} {train_r2:<20.4f} {test_r2:<20.4f}")
    print("="*60)
    
    # Cross-validation score
    print("\n🔄 Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse_scores = np.sqrt(-cv_scores)
    
    print(f"   ✅ 5-Fold CV RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std():.2f})")
    
    # Feature importance
    print("\n🎯 Top 10 Most Important Features:")
    print("-"*60)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:<30} {row['importance']:.4f}")
    
    # Step 9: Save the model
    print(f"\n💾 Saving model to {model_file}...")
    
    # Package model with metadata
    model_package = {
        'model': model,
        'feature_columns': list(X.columns),
        'target_column': target_column,
        'label_encoders': label_encoders,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'feature_importance': feature_importance.to_dict('records'),
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse_mean': cv_rmse_scores.mean(),
            'cv_rmse_std': cv_rmse_scores.std()
        },
        'training_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target_min': float(y.min()),
            'target_max': float(y.max()),
            'target_mean': float(y.mean())
        }
    }
    
    try:
        joblib.dump(model_package, model_file)
        print(f"   ✅ Model saved successfully!")
    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
        return
    
    # Step 10: Display sample predictions
    print("\n📋 Sample Predictions (first 5 test samples):")
    print("="*60)
    print(f"{'Actual':<15} {'Predicted':<15} {'Error':<15}")
    print("-"*60)
    
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_test_pred[i]
        error = abs(actual - predicted)
        print(f"{actual:<15.2f} {predicted:<15.2f} {error:<15.2f}")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"   Model file: {model_file}")
    print(f"   Test R² Score: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"\n   You can now use:")
    print(f"   - python model_predict.py (for single predictions)")
    print(f"   - python model_batch_predict.py (for batch predictions)")
    print("="*60)

if __name__ == "__main__":
    train_model()