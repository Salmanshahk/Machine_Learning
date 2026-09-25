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

print(f"Odds Ratio")

print("Odds Ratio = e^(coefficient)")

print(" > 1 : Feature INCREASES chance of passing")
print(" < 1 : Feature DECREASES chance of passing")
print(" = 1 : No effect\n")

slope = model.coef_[0]
print(f"Slope: ", slope)

odds_ratio = np.exp(model.coef_[0])
print(f"Odds Ratio: ", odds_ratio)

for feature,odds in zip(X.columns,odds_ratio):
    direction = '(+) Increases chance of passing' if odds > 1 else '(-) Decreases chance of passing'
    print(f" {feature:<20} odds_ratio = {odds:.4f} {direction} chance of passing")

print(f"\n\n[PROBABILITY OF SUCCESS & FAILURE")
print("-" * 60)
print(f"{'Sample': >6} | {'P(Fail'} :>8 | {'Predicted': >9} | {'Actual': >6}")
print("-" * 60)

for i in range(len(Y_test)):
    p_fail = y_pred_proba[i][0]
    p_pass = y_pred_proba[i][1]
    pred = y_pred[i]
    actual = Y_test.iloc[i]
    marker = "[ok]" if pred == actual else "[fail]"
    print(f" {i+1:4} | {p_fail:.7f} | {p_pass:.7f} | {'Pass' if pred==1 else 'Fail':9} | {'Pass' if actual == 1 else  'Fail':5} {marker} ")










