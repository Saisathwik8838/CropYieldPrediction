"""
Crop Yield Prediction Web Application
Flask-based web portal for data upload, model training, and predictions
"""
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import os
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing model functions
import sys
sys.path.append('.')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
Path('uploads').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_status():
    """Check if model exists and get its info"""
    model_file = Path('crop_yield_model.pkl')
    if model_file.exists():
        try:
            model_data = joblib.load(model_file)
            return {
                'exists': True,
                'metrics': model_data.get('metrics', {}),
                'training_info': model_data.get('training_info', {}),
                'target_column': model_data.get('target_column', 'Unknown'),
                'n_features': len(model_data.get('feature_columns', []))
            }
        except:
            return {'exists': False}
    return {'exists': False}

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{plot_url}'

@app.route('/')
def index():
    """Home page with dashboard"""
    model_status = get_model_status()
    
    # Get dataset info if exists
    dataset_file = Path('data/final_dataset.csv')
    dataset_info = {}
    if dataset_file.exists():
        try:
            df = pd.read_csv(dataset_file)
            dataset_info = {
                'exists': True,
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': dataset_file.stat().st_size / (1024 * 1024)
            }
        except:
            dataset_info = {'exists': False}
    else:
        dataset_info = {'exists': False}
    
    return render_template('index.html', 
                         model_status=model_status,
                         dataset_info=dataset_info)

@app.route('/upload')
def upload_page():
    """Data upload page"""
    return render_template('upload.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Handle data file uploads"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    data_type = request.form.get('data_type', 'crop')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Determine save path based on data type
        if data_type == 'crop':
            save_dir = Path('data/Crops')
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / filename
        elif data_type == 'temperature':
            filepath = Path('data/temperature.csv')
        elif data_type == 'rainfall':
            filepath = Path('data/rainfall.csv')
        else:
            return jsonify({'success': False, 'message': 'Invalid data type'})
        
        # Save file
        file.save(filepath)
        
        # Read and validate
        try:
            df = pd.read_csv(filepath)
            return jsonify({
                'success': True,
                'message': f'File uploaded successfully! {len(df)} rows, {len(df.columns)} columns',
                'rows': len(df),
                'columns': list(df.columns)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload CSV or Excel files.'})

@app.route('/combine_data', methods=['POST'])
def combine_data():
    """Combine all uploaded datasets"""
    try:
        # Import and run combine function
        from combine_datasets import combine_datasets
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            combine_datasets()
        
        output = f.getvalue()
        
        # Check if successful
        dataset_file = Path('data/final_dataset.csv')
        if dataset_file.exists():
            df = pd.read_csv(dataset_file)
            return jsonify({
                'success': True,
                'message': 'Dataset combined successfully!',
                'rows': len(df),
                'columns': len(df.columns),
                'output': output
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to create combined dataset',
                'output': output
            })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/train')
def train_page():
    """Model training page"""
    dataset_file = Path('data/final_dataset.csv')
    dataset_exists = dataset_file.exists()
    
    dataset_info = {}
    if dataset_exists:
        try:
            df = pd.read_csv(dataset_file)
            dataset_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
        except:
            pass
    
    return render_template('train.html', 
                         dataset_exists=dataset_exists,
                         dataset_info=dataset_info)

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the ML model"""
    try:
        from model_train import train_model as train_func
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            train_func()
        
        output = f.getvalue()
        
        # Check if model was created
        model_file = Path('crop_yield_model.pkl')
        if model_file.exists():
            model_status = get_model_status()
            return jsonify({
                'success': True,
                'message': 'Model trained successfully!',
                'model_status': model_status,
                'output': output
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed',
                'output': output
            })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/predict')
def predict_page():
    """Prediction page"""
    model_status = get_model_status()
    
    # Get feature info if model exists
    feature_info = {}
    if model_status['exists']:
        try:
            model_data = joblib.load('crop_yield_model.pkl')
            feature_info = {
                'categorical': model_data.get('categorical_features', []),
                'numeric': model_data.get('numeric_features', []),
                'all': model_data.get('feature_columns', [])
            }
        except:
            pass
    
    return render_template('predict.html', 
                         model_status=model_status,
                         feature_info=feature_info)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Make single prediction"""
    try:
        from model_predict import predict_from_dict
        
        input_data = request.json
        prediction = predict_from_dict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'input_data': input_data
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch prediction file upload"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Read uploaded file
            df = pd.read_csv(file)
            
            # Make predictions
            from model_batch_predict import predict_from_dataframe
            predictions = predict_from_dataframe(df)
            
            # Add predictions to dataframe
            model_data = joblib.load('crop_yield_model.pkl')
            target_col = model_data['target_column']
            
            df[f'predicted_{target_col}'] = predictions
            
            # Save to file
            output_file = Path('data/batch_predictions.csv')
            df.to_csv(output_file, index=False)
            
            # Create summary stats
            stats = {
                'count': len(predictions),
                'mean': float(np.mean(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'std': float(np.std(predictions))
            }
            
            return jsonify({
                'success': True,
                'message': f'Predictions completed for {len(predictions)} rows',
                'stats': stats,
                'preview': df.head(10).to_dict('records')
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files"""
    file_path = Path('data') / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/analytics')
def analytics_page():
    """Analytics and visualization page"""
    model_status = get_model_status()
    
    plots = {}
    
    if model_status['exists']:
        try:
            model_data = joblib.load('crop_yield_model.pkl')
            
            # Feature importance plot
            feature_imp = model_data.get('feature_importance', [])
            if feature_imp:
                top_features = sorted(feature_imp, key=lambda x: x['importance'], reverse=True)[:10]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                features = [f['feature'] for f in top_features]
                importances = [f['importance'] for f in top_features]
                
                ax.barh(features, importances, color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance')
                ax.invert_yaxis()
                
                plots['feature_importance'] = create_plot_base64(fig)
            
            # Model metrics plot
            metrics = model_data.get('metrics', {})
            if metrics:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                metric_names = ['RMSE', 'MAE', 'R² Score']
                train_vals = [metrics.get('train_rmse', 0), 
                            metrics.get('train_mae', 0),
                            metrics.get('train_r2', 0)]
                test_vals = [metrics.get('test_rmse', 0),
                           metrics.get('test_mae', 0),
                           metrics.get('test_r2', 0)]
                
                x = np.arange(len(metric_names))
                width = 0.35
                
                ax.bar(x - width/2, train_vals, width, label='Training', color='lightblue')
                ax.bar(x + width/2, test_vals, width, label='Testing', color='coral')
                
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Metrics')
                ax.set_xticks(x)
                ax.set_xticklabels(metric_names)
                ax.legend()
                
                plots['metrics'] = create_plot_base64(fig)
                
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    return render_template('analytics.html', 
                         model_status=model_status,
                         plots=plots)

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'model': get_model_status(),
        'dataset': {
            'exists': Path('data/final_dataset.csv').exists()
        }
    })

if __name__ == '__main__':
    print("🌾 Starting Crop Yield Prediction Web Application...")
    print("📊 Access at: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)