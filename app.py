from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pickle
import joblib
import pandas as pd
import pandas as pd
from joblib import load
import os
import google.generativeai as genai



app = Flask(__name__)

import joblib
model = joblib.load('model.pkl')  # or whichever model you finalized
scaler = joblib.load('sc.sav')


genai.configure(api_key="AIzaSyDRYDIeKNxnLSF2QQHCMQEoeVWtD3bPmsQ")  
model = genai.GenerativeModel("gemini-1.5-flash")
chat_session = model.start_chat(history=[])


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")

    prompt = f"""You are BigMart Sales Forecasting Assistant. Your job is to help users understand how this system works. 
    The system predicts sales for retail products using machine learning models based on input features like item type, MRP, outlet type, etc.
    
    User: {user_message}
    Assistant:"""

    try:
        response = chat_session.send_message(prompt)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": "Sorry, I had an issue responding. Please try again."})


@app.route('/dashboard')
def dashboard():
    try:
        df = pd.read_csv('predictions.csv')  # Load prediction data
        df.columns = [col.strip() for col in df.columns]  # Clean up whitespace just in case
        df['Item_Type'] = df['Item_Type'].str.strip().str.title()
        #df = df.rename(columns={df.columns[-1]: "Predicted_Sales"})  # Make sure last column has the right name
        df.columns = ['Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Type', 'Predicted_Sales']

        grouped = df.groupby('Item_Type')['Predicted_Sales'].sum().reset_index()
        grouped.columns = ['Item_Type', 'Predicted_Sales']

        # Pass this to template

        predictions = grouped.to_dict(orient='records')

        return render_template('dashboard.html', predictions=predictions)
    except Exception as e:
        return f"Error loading predictions: {str(e)}"
    

@app.route('/add_record', methods=['POST'])
def add_record():
    # Get form data
    item_type = request.form['item_type']
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_type = request.form['outlet_type']

    # Add default or placeholder values for other required fields
    new_data = pd.DataFrame([{
        'Item_Type': item_type,
        'Item_MRP': item_mrp,
        'Outlet_Establishment_Year': outlet_establishment_year,
        'Outlet_Type': outlet_type,
    }])

    try:
    # Try to preprocess and predict
     processed_data = preprocess_new_record(new_data)
     prediction = model.predict(processed_data)[0]
    except Exception as e:
     return f"Prediction error: {str(e)}"

    # write to CSV if prediction succeeded
    with open('predictions.csv', 'a') as f:
     f.write(f"{item_type},{item_mrp},{outlet_establishment_year},{outlet_type},{prediction}\n")

    # Save to file/database
    file_exists = os.path.isfile('predictions.csv')
    
    row = pd.DataFrame([{
    'Item_Type': item_type,
    'Item_MRP': item_mrp,
    'Outlet_Establishment_Year': outlet_establishment_year,
    'Outlet_Type': outlet_type,
    'Predicted_Sales': prediction
    }])

    # Append to CSV file with headers if file doesn't exist
    try:
      existing = pd.read_csv('predictions.csv')
      updated = pd.concat([existing, row], ignore_index=True)
    except FileNotFoundError:
      updated = row  # First entry

    # Save updated DataFrame
    updated.to_csv('predictions.csv', index=False)

    return redirect('/dashboard')


def preprocess_new_record(new_data):
    # One-hot encode new data
    new_data_encoded = pd.get_dummies(new_data)

    # Load the feature names used during model training
    feature_names = pickle.load(open("feature_names.pkl", "rb"))

    # Add missing columns and set them to 0
    for col in feature_names:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0

    # Remove any unexpected columns
    new_data_encoded = new_data_encoded[feature_names]

    # Scale the aligned data
    scaler = load("sc.sav")
    new_data_scaled = scaler.transform(new_data_encoded)

    return new_data_scaled



# Load trained model and scaler safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = joblib.load("sc.sav")
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model or scaler file not found. Ensure 'model.pkl' and 'sc.sav' exist.")
    exit(1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template("predict.html")

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    try:
        # Extracting form inputs
        form_data = request.form
        item_fat_content = form_data.get('item_fat_content')
        item_visibility = float(form_data.get('item_visibility'))
        item_type = form_data.get('item_type')  # no int()
        item_mrp = float(form_data.get('item_mrp'))
        outlet_establishment_year = int(form_data.get('outlet_establishment_year'))
        outlet_size = form_data.get('outlet_size')
        outlet_location_type = form_data.get('outlet_location_type')
        outlet_type = form_data.get('outlet_type')

        # Ensure no missing values
        if None in [item_fat_content, item_visibility, item_type, item_mrp,
                    outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]:
            return jsonify({'error': 'Missing input values'}), 400

        # Create DataFrame with user input
        df_input = pd.DataFrame([[
            item_fat_content, item_visibility, item_type,
            item_mrp, outlet_establishment_year, outlet_size,
            outlet_location_type, outlet_type
        ]], columns=['Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                     'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
                     'Outlet_Location_Type', 'Outlet_Type'])

        # One-hot encode categorical features
        categorical_features = ['Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
        df_encoded = pd.get_dummies(df_input, columns=categorical_features)

        # Add missing columns efficiently (Fixes Fragmentation Warning)
        missing_cols = list(set(feature_names) - set(df_encoded.columns))
        df_missing = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
        df_encoded = pd.concat([df_encoded, df_missing], axis=1)

        # Reorder columns to match training
        df_encoded = df_encoded[feature_names]

        # Convert to NumPy array before scaling (Fixes Sklearn Feature Names Warning)
        X_std = scaler.transform(df_encoded.to_numpy())

        # Predict
        prediction = model.predict(X_std)

        return jsonify({'prediction': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=9457)
