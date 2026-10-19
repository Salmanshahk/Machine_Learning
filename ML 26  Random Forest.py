import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    learning_curve
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    f1_score
)

df = read_csv('')
print(f" 1. Dataset loaded : {df.shape[0]} rows x {df.shape[1]} Columns")
print(" columns :",list(df.columns[:6]))
print(df.head(3).to_string())

