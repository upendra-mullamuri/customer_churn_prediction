"""
Generates a synthetic dataset that mirrors the schema and statistical
patterns of the well-known IBM Telco Customer Churn dataset.

Why synthetic? This environment has no internet access to pull the live
Kaggle file. The generator below reproduces realistic feature distributions
and, importantly, bakes in genuine churn-driving relationships (month-to-month
contracts, high monthly charges, low tenure, no tech support, fiber optic
service, electronic check payment) so the EDA and modeling below reflect
real signal, not noise.

If you have internet access, you can instead download the real dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
and drop it in as data/telco_churn.csv with the same column names to run
this project unchanged.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7000

genders = np.random.choice(["Male", "Female"], N)
senior = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = np.random.gamma(shape=2.0, scale=15, size=N).astype(int).clip(0, 72)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def dependent_service(base_col, p_yes=0.4):
    return np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], N, p=[p_yes, 1 - p_yes])
    )

online_security = dependent_service("online_security", 0.29)
online_backup = dependent_service("online_backup", 0.34)
device_protection = dependent_service("device_protection", 0.34)
tech_support = dependent_service("tech_support", 0.29)
streaming_tv = dependent_service("streaming_tv", 0.38)
streaming_movies = dependent_service("streaming_movies", 0.39)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

base_charge = np.where(internet_service == "Fiber optic", 70,
              np.where(internet_service == "DSL", 45, 20))
addon_count = (
    (online_security == "Yes").astype(int) + (online_backup == "Yes").astype(int) +
    (device_protection == "Yes").astype(int) + (tech_support == "Yes").astype(int) +
    (streaming_tv == "Yes").astype(int) + (streaming_movies == "Yes").astype(int)
)
monthly_charges = (base_charge + addon_count * 5 + np.random.normal(0, 5, N)).clip(18, 120).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).clip(0, None).round(2)

# ---- Bake in realistic churn drivers via a logistic latent score ----
score = (
    -0.04 * tenure
    + 0.02 * monthly_charges
    + 1.1 * (contract == "Month-to-month")
    + 0.5 * (internet_service == "Fiber optic")
    + 0.5 * (payment_method == "Electronic check")
    - 0.6 * (tech_support == "Yes")
    - 0.5 * (partner == "Yes")
    - 0.3 * (dependents == "Yes")
    + 0.3 * (senior == 1)
    - 2.0
)
prob_churn = 1 / (1 + np.exp(-score))
churn = np.random.binomial(1, prob_churn)

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": np.where(churn == 1, "Yes", "No"),
})

df.to_csv("/home/claude/customer_churn/data/telco_churn.csv", index=False)
print(df.shape)
print(df["Churn"].value_counts(normalize=True))
