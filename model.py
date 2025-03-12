import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai  # Google's Gemini API
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set up Gemini API key securely
GEMINI_API_KEY = "YOURAPIKEY"
genai.configure(api_key=GEMINI_API_KEY)

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Separate features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fairness mitigation using reweighting
def mitigate_bias(y):
    return np.where(y == 1, 1.5, 1)  # Adjusting weights for fairness

# Apply bias mitigation
sample_weights = mitigate_bias(y_train)

1
# Train a Random Forest model with sample weights
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# Using Gemini API to explain model decisions
def explain_decision(input_data):
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    response = model_gemini.generate_content(
        f"Explain the following credit scoring decision based on user details: {input_data}"
    )
    return response.text

# Example explanation
sample_input = X_test.iloc[0].to_dict()
explanation = explain_decision(sample_input)
print("\nDecision Explanation from Gemini API:")
print(explanation)

# Save the model for future use
joblib.dump(model, "credit_scoring_model.pkl")
print("\nModel saved as 'credit_scoring_model.pkl'")
