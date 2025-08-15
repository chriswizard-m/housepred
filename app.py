from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Load the preprocessor and model ---
try:
    preprocessor = joblib.load('predmod/preprocessor.pkl')
    model = load_model('predmod/house_price_model.h5')
except FileNotFoundError:
    print("Error: 'preprocessor.pkl' or 'house_price_model.h5' not found.")
    print("Please run the 'Step 1' script first to generate these files.")
    exit()

# --- Define the API endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the house price based on input features.
    
    The expected JSON format is:
    {
        "Gr Liv Area": 1500,
        "Bedroom AbvGr": 3,
        "Overall Qual": 7,
        "Yr Sold": 2022,
        "Bldg Type": "1Fam"
    }
    """
    try:
        data = request.get_json(force=True)
        
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])
        
        # Preprocess the input data using the loaded preprocessor
        input_processed = preprocessor.transform(input_df)
        
        # Make the prediction
        prediction_array = model.predict(input_processed)
        prediction = float(prediction_array[0][0])
        
        return jsonify({
            'predicted_price': prediction
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

# --- Run the application ---
if __name__ == '__main__':
    # You can change the host and port if needed
    app.run(host='0.0.0.0', port=5000)