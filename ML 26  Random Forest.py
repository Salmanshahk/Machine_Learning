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

TARGEL_COL = df['diagnosis']

print("\n Class Distribution")

print(TARGEL_COL.value_counts().to_string())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(13,8))

counts = TARGEL_COL.value_counts()
bars = axes[0].bar(
    counts.index, 
    counts.values,
    color='darkgreen',
    edgecolor='k',
    width = 0.45)
    
for bar , v in zip(bars,counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., v+3,
    f"{v}",ha='center',fontsize =12,fontweight='bold')

axes[0].set_title("Class Distribution",fontsize =12,fontweight='bold')
axes[0].set_xlabel("diagnisis")
axes[0].set_ylabel("count")

plot_col = "mean radius"

groups = [df[df['diagnosis']==ip1][plot_col].values for ip1 in counts.index]
bp = axes[1].boxplot(
    groups, 
    labels = counts.index, 
    patch_artist=True,
    boxprops = dict(
        facecolor='green',
        edgecolor = 'k'),
        medianprops = dict(
        color = 'red',
        linewidth = 2.5))
axes[1].set_title(f"{plot_col} by Diagnosis",fontsize =12,fontweight='bold')
axes[1].set_ylabel(plot_col)
plt.suptitle(
    "EDA Overview",
    fontsize=15,
    fontweight='bold')

plt.tight_layout()
#plt.show()


plt.figure(figsize=(12,9))
corr = df[numeric_cols[:10]].corr()
mask = np.triu(np.ones_like(corr,dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    cmap='coolwarm',
    annot = True,
    fmt='.2f',
    square=True,
    linewidths = 0.5,
    annot_kws={"size":9})
plt.title(
    "Corr Heatmap",
    fontsize =12,
    fontweight='bold')
plt.show()


















