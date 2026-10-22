import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Create dataset
data = {
    "Age": [23, 30, 42, 22, 30, 50, 28, 40, 32, 27],
    "MonthlyCharges": [500, 1200, 800, 1500, 900, 700, 1400, 600, 1100, 1300],
    "Tenure": [24, 6, 36, 1, 18, 48, 3, 60, 12, 4],
    "SupportCalls": [1, 8, 0, 10, 2, 0, 7, 1, 5, 9],
    "ContractType": [
        "Yearly",
        "Monthly",
        "Yearly",
        "Monthly",
        "Yearly",
        "Yearly",
        "Monthly",
        "Yearly",
        "Monthly",
        "Monthly",
    ],
    "Churn": [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
}

df = pd.DataFrame(data)
print("Initial DataFrame Head:")
print(df.head())

# Encode categorical variable
le = LabelEncoder()
df["ContractType"] = le.fit_transform(df["ContractType"])
print("\nEncoded DataFrame Head:")
print(df.head())

# Define Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Model
model = XGBClassifier(
    n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, zero_division=0),
)

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": importance}
).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(
    feature_importance["Feature"], feature_importance["Importance"], color="g"
)
plt.xlabel("Feature")
plt.ylabel("Importance Features")
plt.title("Feature Importance using XGBoost")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



new_customer = pd.DataFrame({
    "Age": [29],
    "MonthlyCharges": [1450],
    "Tenure": [2],
    "SupportCalls": [8],
    "ContractType": [0] # Monthly
})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Customer will churn")
else:
    print("Customer will not churn")
    