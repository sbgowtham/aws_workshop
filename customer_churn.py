#go to Amazon SageMaker AI -> create a note book with full s3 access
#(predicting customer churn) and the key AWS services utilized (Amazon Aurora for data storage and Amazon SageMaker for machine learning).

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# Step 1: Define S3 Bucket and File Path
# -------------------------------

s3_bucket = 'dms-data-bucket-sbg'
s3_key = 'test/Customers/LOAD00000001.csv'  # Update with your file name and path

# -------------------------------
# Step 2: Create S3 Client and Download CSV
# -------------------------------

# Initialize S3 client
s3 = boto3.client('s3')

# Download the CSV file from S3
try:
    s3.download_file(s3_bucket, s3_key, 'LOAD00000001.csv')
    print(f"Successfully downloaded 'LOAD00000001.csv' from s3://{s3_bucket}/{s3_key}.")
except Exception as e:
    print(f"Error downloading file: {e}")
    raise

# -------------------------------
# Step 3: Define Column Names
# -------------------------------

# Define the column names as per your Aurora Customers table
column_names = ['CustomerID', 'Name', 'Age', 'SubscriptionType', 'MonthlySpend', 'Churned']

# -------------------------------
# Step 4: Load CSV into pandas DataFrame with Specified Columns
# -------------------------------

try:
    # Load the CSV without headers and assign column names
    df = pd.read_csv('customers.csv', header=None, names=column_names)
    print("Data loaded successfully with specified column names.")
except Exception as e:
    print(f"Error loading CSV into DataFrame: {e}")
    raise

# -------------------------------
# Step 5: Explore the Data
# -------------------------------

print("\nFirst 5 rows of the DataFrame:")
print(df.head())

print("\nColumns in the DataFrame:")
print(df.columns.tolist())

# -------------------------------
# Step 6: Data Preparation
# -------------------------------

# Handle missing values by dropping rows with any null values
df = df.dropna()
print("\nAfter dropping missing values:")
print(df.shape)

# Initialize LabelEncoder
le = LabelEncoder()

# Check if 'SubscriptionType' exists and encode it
if 'SubscriptionType' in df.columns:
    print("\n'SubscriptionType' column found. Proceeding with encoding.")
    df['SubscriptionType'] = le.fit_transform(df['SubscriptionType'])
else:
    print("\n'SubscriptionType' column not found. Dropping this feature from the dataset.")
    # If 'SubscriptionType' is essential, you might want to handle it differently
    # For simplicity, we'll exclude it from features
    column_names.remove('SubscriptionType')

# Define features and target variable
features = ['Age', 'MonthlySpend']
if 'SubscriptionType' in df.columns:
    features.append('SubscriptionType')

X = df[features]
y = df['Churned']

print(f"\nFeatures used for training: {features}")
print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# -------------------------------
# Step 7: Train the Machine Learning Model
# -------------------------------

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("\nModel training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Save the Trained Model to S3
# -------------------------------

# Define the local filename and S3 key for the model
model_filename = 'churn_model.joblib'
model_s3_key = 'models/churn_model.joblib'  # Adjust the path as needed

# Save the model locally using joblib
try:
    joblib.dump(model, model_filename)
    print(f"\nModel saved locally as '{model_filename}'.")
except Exception as e:
    print(f"Error saving the model locally: {e}")
    raise

# Upload the model to S3
try:
    s3.upload_file(model_filename, s3_bucket, model_s3_key)
    print(f"Model uploaded to S3 at s3://{s3_bucket}/{model_s3_key}.")
except Exception as e:
    print(f"Error uploading the model to S3: {e}")
    raise
