import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Path to your training data and new data
training_data_path = r"C:\Users\raksh\OneDrive\Desktop\Rakshita Sataraddi\adverk\adverse_events.csv"
new_data_path = r"C:\Users\raksh\OneDrive\Desktop\Rakshita Sataraddi\adverk\ASR.csv"  # Change this to your new data's path

# Load the training data
training_data = pd.read_csv(training_data_path)

# List of categorical features as they appear in both datasets
categorical_features = [
    "Manufacturer Name",  # Matches in both datasets
    "Initial Report Flag",  # Matches in both datasets
    "Product Code",  # Matches in both datasets
    "Brand Name",  # Matches in both datasets
    "Model Number",  # Matches in both datasets
    "Catalog Number"  # Matches in both datasets
]

# Ensure the dataset contains the categorical features
missing_features = [feature for feature in categorical_features if feature not in training_data.columns]
if missing_features:
    raise ValueError(f"Missing features in training dataset: {missing_features}")

# Fill missing values in the categorical features
training_data[categorical_features] = training_data[categorical_features].fillna("Unknown")

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit the encoder on the categorical features of the training data
encoder.fit(training_data[categorical_features])

# Save the encoder to a file for later use
encoder_file_path = r"C:\Users\raksh\01\encoder.joblib"
joblib.dump(encoder, encoder_file_path)
print(f"Encoder has been successfully fitted and saved as '{encoder_file_path}'")

# Now let's load and use the encoder for a new dataset
def transform_new_data(new_data_path):
    # Load new data (replace with your actual file path)
    new_data = pd.read_csv(new_data_path)

    # Print the columns of both datasets to compare
    print("Columns in new data:", new_data.columns)
    print("Columns in training data:", training_data.columns)

    # Ensure the new data contains the same categorical features
    missing_features = [feature for feature in categorical_features if feature not in new_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in new dataset: {missing_features}")
    
    # Fill missing values for categorical features in new data
    new_data[categorical_features] = new_data[categorical_features].fillna("Unknown")
    
    # Load the previously saved encoder
    encoder = joblib.load(encoder_file_path)
    print("Encoder loaded successfully!")

    # Transform the categorical features using the loaded encoder
    transformed_data = encoder.transform(new_data[categorical_features])

    # Convert transformed data to DataFrame with appropriate column names
    transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out(categorical_features))

    return transformed_df

# Example of transforming new data (You can call this function where needed)
transformed_new_data = transform_new_data(new_data_path)
print(transformed_new_data.head())  # Print the transformed data for verification


transformed_new_data.to_csv("transformed_new_data.csv", index=False)
print("Transformed data saved as 'transformed_new_data.csv'")
