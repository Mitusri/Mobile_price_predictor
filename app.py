from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class PythonAnywherePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.price_ranges = {
            0: "Low Cost (0-5000)",
            1: "Medium Cost (5000-10000)", 
            2: "High Cost (10000-15000)",
            3: "Very High Cost (15000+)"
        }
        
    def train_and_save(self):
        """Train the model and save it"""
        print("Training mobile price prediction model...")
        
        # Load data
        train_df = pd.read_csv('train.csv')
        X = train_df.drop('price_range', axis=1)
        y = train_df['price_range']
        
        # Split and scale
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save
        self.feature_names = X.columns.tolist()
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open('pa_mobile_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        accuracy = self.model.score(X_val_scaled, y_val)
        print(f"Model trained! Accuracy: {accuracy:.2%}")
        return accuracy
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('pa_mobile_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, features_dict):
        """Predict price range from features dictionary"""
        if self.model is None:
            if not self.load_model():
                self.train_and_save()
        
        # Prepare input
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[self.feature_names]
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        return {
            'price_range': int(prediction),
            'price_category': self.price_ranges[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {str(k): float(v) for k, v in enumerate(probabilities)}
        }

# Initialize predictor
predictor = PythonAnywherePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert string values to appropriate types
        features = {}
        for key, value in data.items():
            if key in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
                features[key] = int(value)
            else:
                features[key] = float(value)
        
        result = predictor.predict(features)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/feature_ranges')
def get_feature_ranges():
    """Get valid ranges for features"""
    train_df = pd.read_csv('train.csv')
    X = train_df.drop('price_range', axis=1)
    
    ranges = {}
    for col in X.columns:
        ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'type': 'binary' if col in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'] else 'numeric'
        }
    
    return jsonify(ranges)

@app.route('/health')
def health_check():
    """Health check endpoint for PythonAnywhere"""
    return jsonify({'status': 'healthy', 'message': 'Mobile Price Predictor is running!'})

if __name__ == '__main__':
    # Train model on startup
    if not predictor.load_model():
        predictor.train_and_save()
    
    print("🌐 Starting Mobile Price Predictor for PythonAnywhere...")
    app.run(debug=False, host='0.0.0.0', port=5000) 