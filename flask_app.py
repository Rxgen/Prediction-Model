from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('trend_prediction_model.pkl')

# Define feature list
features = ['bsr_movement', 'price_change', 'review_growth', 'rating_change', 'listing_age_days', 'new_product_flag']

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Trend Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON input
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure all features are present
        for feature in features:
            if feature not in df:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Predict probability
        probability = model.predict_proba(df[features])[:, 1][0]
        result = {
            'trend_probability': round(probability * 100, 2)
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
