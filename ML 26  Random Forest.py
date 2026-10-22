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

# Pre-processing

X_raw = df.drop(columns=['diagnosis'])
Y_raw = df['diagnosis']

le = LabelEncoder()

y = le.fit_transform(Y_raw)
label_map = dict(zip(le.classes_, le.transform(le.classes_)))

print(F"Label Mapping:{label_map}")

X = X_raw.select_dtypes(include=[np.number])
feature_names = list(X.columns)

print(f"Total Features: {len(feature_names)}")

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n Train {len(X_train)} samples | Test {len(y_train)} samples")

scaler = StandardScaler()
X_train_SC = scaler.fit_transform(X_train)
X_test_SC = scaler.transform(X_test)
X_all_SC = scaler.transform(X)

param_grid = {
    "n_estimators" :[100,200],
    "max_depth":[None,5,10],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        n_jobs = -1),
    param_grid = param_grid,
    cv=5,
    scoring = 'accuracy',
    n_jobs= -1,
    verbose=0
)

grid.fit(X_train_SC,y_train)

best_params = grid.best_params_

print(f"Best params :{best_params}")
print(f"Best CV Acc : {grid.best_score_}")
print(f"")
best_rf = grid.best_estimator_


# K FOLD CROSS VALIDATION
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42)

cv_scores = cross_val_score(
    best_rf,
    X_all_SC,
    y,
    cv = cv,
    scoring='accuracy')

print(f"\n 5 fold cv Scores: {np.round(cv_scores,4)}")

print(f"\n Mean : {cv_scores.mean(),4} | std {cross_val_score.std():4}")

plt.figure(figsize=(10,6))
plt.bar(
    range(1,6),
    cv_scores,
    color='#42a5f5',
    edgecolor='black',
    alpha=0.85)

plt.axhline(
    cv_scores.mean(),
    color = 'r',
    lw = 2,
    linestyle ='--',
    label = f'Mean Acc: {cv_scores.mean():4}')
plt.title("K FOLD VALIDATION")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.87,1.01)
plt.legend()
plt.tight_layout()
plt.show()

# Model Evaluation
best_rf.fit(X_train_SC,y_train)

y_pred = best_rf.predict(X_test_SC)
y_proba = best_rf.predict_proba(X_test_SC)[:,1]

test_acc = accuracy_score(y_test,y_pred)
test_auc = roc_auc_score(y_test,y_proba)

print("\n+*60")

print(f"Test Accuracy : {test_acc:.4f} ({test_auc*100:.2f}%)")

print(f"\n ROC-AUC Score : {roc_auc:.4f}")

print(classification_report(y_test,y_pred,target_names=le.classes_))

# Feature Importance

importances = best_rf.feature_importances_

sorted_idx = np.argsort(importances)[::-1]

top_n=15

print(f"\n Top {top_n} Feature Importances (MDI)")
for rank, idx in enumerate(sorted_idx[:top_n],1):
    print(f" {rank:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f}")
    
colors = plt.cm.plasma(np.linspace(0.15,0.85,top_n))

plt.figure(figsize=(10,7))
bplt.barh(
    range(top_n),
    importances[sorted_idx[:top_n]][::-1],
    color=colors,
    align = 'center',
    edgecolor='grey')
y_tick_fe = np.array(feature_names[sorted_idx[:top_n]])[::-1]
plt.yticks(
    range(top_n),
    y_tick_fe,
    fontsize=9)
plt.xlabel("Mean Decrease Impurity (MDI)",fontsize=12,fontweight='bold')
plt.title(f"Top {top_n} Features",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.show()

# Confusion Matrix

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt ="d",
    cmap='Blues',
    xticklabels = le.classes_,
    yticklabels= le.classes_,
    linewidths=0.5,
    annot_kws={"size":18,"weight":"bold"},
    linecolor = 'k',
    cbar=False)
    
plt.title(
    "Confusion Matrix - Test Set",
    fontsize = 13,
    fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

fpr,tpr,_ = roc_curve(y_test,y_proba)

plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    color='#2196f3',
    lw=2.5,
    label=f"Random Forest AUC Curve = {roc_auc:.3f})")
plt.plot(
    [0,1],
    [0,1],
    color='k',
    lw=2,
    linestyle='--',
    label="Random Baseline")
plt.fill_between(
    fpr,
    tpr,
    alpha=0.12,
    color="darkorange"
)
plt.xlabel(
    'False positive rate(1-specificity)',
    fontsize=12)
plt.ylabel("True positive rate")
plt.title(
    f"ROC-AUC Curve",
    fontsize=14,
    fontweight='bold')
plt.legend(loc='lower right',fontsize=10)
plt.grid(True,alpha=0.25)
plt.tight_layout()
plt.show()

# Handling Class Imbalances

print("\n============================================================================================")


rf_balanced = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    random_state=42,
    n_jobs = -1
    )

rf_balanced.fit(X_train_sc,y_train)

y_pred_bal = rf_balanced.predict(X_test_sc)
acc_balanced = accuracy_score(
    y_test,
    y_pred_bal)
f1_balanced = f1_score(
    y_test,
    y_pred_bal)

print(f"    Accuracy (Balanced)  : {acc_balanced:.4f}")

print(f"    Micro F1 (Balanced)  : {f1_balanced:.4f}")

print(f"    Accuracy (Original)  : {test_acc:.4f}")


# SKLearn Pipeline => Preprocessing + Model in one Project

pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ('rf',RandomForestClassifier(
        **best_params,
    class_weight='balanced',
    random_state=42,
    n_jobs = -1
    ))
])

pipeline.fit(X_train,y_train)

pipeline_acc = accuracy_score(y_test,pipeline.predict(X_test))

print(f"Pipeline test Accuracy: {pipeline_acc:.4f}")

# OOB Out Of Bag

rf_oob = RandomForestClassifier(
        **best_params,
    class_weight='balanced',
    oob_score= True,
    random_state=42,
    n_jobs = -1
    )

rf_oob.fit(X_all_SC,y)

print(f"  OOB Score (Whole Dataset) :{rf_oob.oob_score_:.4f}")

# Learning Curve

train_sizes,train_scores,val_scores=learning_curve(
    best_rf,
    X_all_SC,
    y,
    train_sizes=np.linspace(0.1,1.0,10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    )

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scoress.mean(axis=1)
val_std = val_scoress.std(axis=1)

plt.figure(figsize=(10,6))

plt.plot(
    train_sizes,
    train_mean,
    color='#1E88E5',
    marker='o-',
    markersize=5,
    linewidth=2.5,
    label="Training Accuracy")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    color='#1E88E5')

plt.plot(
    train_sizes,
    val_mean,
    color = '#ff6f00',
    marker='s-',
    markersize=5,
    linewidth=2.5,
    label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.15,
    color = '#ff6f00')

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title(
    "Learning Curve - Bias vs Variance Analysis",
    fontsize=13,
    fontweight='bold')
plt.legend(loc="lower right",fontsize=10)
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()