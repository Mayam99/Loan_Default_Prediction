# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your cleaned dataset
data = pd.read_csv("cleaned_dataset.csv")

# --- Update these with your real column names ---
X = data.drop("Loan_Default", axis=1)
y = data["Loan_Default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as logistic_model.pkl")
