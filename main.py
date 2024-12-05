from json import encoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from typing import List
import logging
import numpy as np

encoder = joblib.load("adverse_events.csv")  # Encoder for categorical data
model = joblib.load('model.joblib')  # Pre-trained machine learning model

# Initialize FastAPI
app = FastAPI()

# Define the input data structure
class EventData(BaseModel):
    exemption_number: str
    manufacturer_registration_number: int
    manufacturer_name: str
    report_id: str
    date_of_event: str  # Optional, but keep it as string if provided
    manufacturer_aware_date: str
    device_problem_codes: int
    report_year: int
    report_quarter: int
    initial_report_flag: str
    device_id: int
    product_code: str
    brand_name: str
    model_number: int
    catalog_number: str
    implant_available_for_evaluation: float
    implant_returned_to_manufacturer: float

@app.post("/predict/")
def predict(event_data: EventData):
    try:
        # Convert input data to a DataFrame for processing
        data = pd.DataFrame([event_data.dict()])

        # Fill missing values (if necessary, based on your dataset's requirements)
        data.fillna(value={"model_number": "Unknown", "catalog_number": "Unknown", "brand_name": "Unknown"}, inplace=True)

        # Encode categorical variables (assumes encoder is already fitted)
        categorical_features = [
            "manufacturer_name",
            "device_problem_codes",
            "initial_report_flag",
            "device_id",
            "product_code",
            "brand_name",
            "model_number",
            "catalog_number",
        ]
        for col in categorical_features:
            data[col] = encoder.transform(data[[col]])  # Encode categorical features

        # Prepare the feature vector
        feature_vector = np.array(data.values).reshape(1, -1)

        # Predict using the pre-trained model
        prediction = model.predict(feature_vector)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}
    

    
    # One-hot encode the manufacturers (or other features used in your model)
   ### manufacturer_data = mlb.transform([event_data.manufacturers])
    ###input_data.update({label: manufacturer_data[0][i] for i, label in enumerate(mlb.classes_)})

    # Convert input data to a DataFrame
    ###input_df = pd.DataFrame(input_data)

    # Make predictions using the model
    ###prediction = model.predict(input_df)

    # Return the prediction result as a JSON response
    ###return {"predicted_event_type": prediction[0]}

# To run the app, use the command below:
# uvicorn main:app --reload
