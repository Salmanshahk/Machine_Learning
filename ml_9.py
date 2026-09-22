import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("E:\Data Science, ML & AI\Varun & Aashique\Machine_Leaning\student_performance_dataset.csv")

print(df.head())

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

print(f"\nColumns : {list(df.columns)}")

print("\nFirst 5 rows")

print(df.head().to_string())

print("\nstatistics")

print(df.describe(include='all').to_string())

print(f"\n Target Variable Distribution ")

print(f" Pass (1)  {df['Pass'].sum()} students")
print(f" Fail (0)  {(df['Pass']==0).sum()} students")

X = df[['Study_Hours','Attendance_Percentage','Previous_Score']]
Y = df['Pass']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data Split")

print(f"Training set: {X_train.shape[0]} Samples")
print(f"Testing set: {X_test.shape[0]} Samples")

model = LogisticRegression(random_state=42, max_iter=500)

model.fit(X_train_scaled,Y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print("Model training Result :")

print(" Model Coefficients")

for feature,coef in zip(X.columns,model.coef_[0]):
    print(f" {feature:.<30} {coef:8.4f}")

print(f"    {'Intercept':.<30} {model.intercept_[0]:>8.4f} ")

print(f"\n Accuracy:{accuracy_score(Y_test,y_pred)*100:.2f}%")