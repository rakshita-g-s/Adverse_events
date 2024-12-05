from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
import os
import uvicorn



# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load pre-trained encoder and model
encoder = joblib.load('encoder.joblib')  # Make sure this is a fitted encoder
model = joblib.load('model.joblib')

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

    # Disable protected namespaces to avoid conflicts
    model_config = {
        'protected_namespaces': ()
    }


@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(event_data: EventData):
    try:
        # Convert input data to a DataFrame for processing
        data = pd.DataFrame([event_data.dict()])

        # Fill missing values (if necessary, based on your dataset's requirements)
        data.fillna(value={"model_number": "Unknown", "catalog_number": "Unknown", "brand_name": "Unknown"}, inplace=True)

        # Encode categorical variables using the pre-fitted encoder
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
            if col in data.columns:
                encoded_data = encoder.transform(data[[col]])
                # Join the encoded columns back to the DataFrame
                data = data.join(pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col])))
                data.drop(columns=[col], inplace=True)

        # Prepare the feature vector
        feature_vector = data.values.reshape(1, -1)

        # Predict using the pre-trained model
        prediction = model.predict(feature_vector)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return {"error": f"Internal Server Error: {str(e)}"}





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default to 8000 if not set
    uvicorn.run(app, host="127.0.0.1", port=port)

