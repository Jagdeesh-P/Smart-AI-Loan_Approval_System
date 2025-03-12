import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)
# ---Loan Approval Dataset ---
print("\nProcessing Loan Approval Dataset...")
try:
    loan_df = pd.read_csv("loan_approval_dataset.csv")
except FileNotFoundError:
    print("Error: loan_approval_dataset.csv not found. Skipping...")
    loan_df = pd.DataFrame()

if not loan_df.empty:
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    for col in ["person_income", "loan_amnt", "loan_int_rate", "credit_score"]:
        loan_df[col] = imputer.fit_transform(loan_df[[col]])
    imputer_cat = SimpleImputer(strategy="most_frequent")
    loan_df["person_education"] = imputer_cat.fit_transform(loan_df[["person_education"]]).ravel()

    # Remove sensitive feature
    loan_df = loan_df.drop(columns=["person_gender"])

    # Normalize numerical data
    scaler = MinMaxScaler()
    numerical_cols = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", 
                      "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
    loan_df[numerical_cols] = scaler.fit_transform(loan_df[numerical_cols])

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    for col in ["person_education", "person_home_ownership", "loan_intent"]:
        encoded = encoder.fit_transform(loan_df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]])
        loan_df = pd.concat([loan_df.drop(columns=[col]), encoded_df], axis=1)

    # Feature engineering
    loan_df["Debt_to_Income"] = loan_df["loan_amnt"] / loan_df["person_income"]
    # Handle string values in previous_loan_defaults_on_file
    loan_df["Has_Defaults"] = loan_df["previous_loan_defaults_on_file"].apply(
        lambda x: 1 if str(x).lower() in ["yes", "true", "defaulted"] else 0
    )
    loan_df = loan_df.drop(columns=["previous_loan_defaults_on_file"])

    # Target variable
    loan_df["loan_status"] = loan_df["loan_status"].astype(int)

    print("Loan Approval Dataset Shape:", loan_df.shape)
    print(loan_df.head())


# --- Save Preprocessed Datasets ---
if not loan_df.empty:
    loan_df.to_csv("preprocessed_loan_approval_dataset.csv", index=False)

print("\nPreprocessing complete! Files saved as preprocessed_*.csv")