# Loan_Approval Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score



CSV_FILE = r"E:\Data Science, ML & AI\DS, ML, DL & AI\ML Datasets\loan_dataset.csv"
TARGET_COL = 'Loan_Status'
df=pd.read_csv(CSV_FILE)

def new_line():
    print("___"*45)

print(df.head())

new_line()
print(f"Loaded '{CSV_FILE}' -> {df.shape[0]} Rows x {df.shape[1]} Columns")

new_line()
print("First 5 rows")

print(df.head())
new_line()
new_line()
new_line()
print("Columns Data types")

print(df.dtypes)

print("Basic Statistics")

print(df.describe(include='all'))
new_line()
print("Missing Values")

print(df.isnull().sum())

fig,axes = plt.subplots(2,3,figsize=(20,12))
fig.suptitle("Exploratory Data Analysis",fontsize = 16,y=1.01)
df[TARGET_COL].value_counts().plot(kind="bar", ax = axes[0,0], color=["cyan","magenta"],edgecolor="k",linewidth=2)
axes[0,0].set_title("Loan Status Distribution")
axes[0,0].set_xlabel("Loan Status")
axes[0,0].set_ylabel("Count")
axes[0,0].tick_params(axis="x",rotation=0)

# Status by Gender
sns.countplot(
    data=df,x='Gender', 
    palette = {'Y':"steelblue",'N':"salmon"}, 
    edgecolor = 'k', 
    hue=TARGET_COL, 
    ax=axes[0,1], 
    linewidth = 0.5)
axes[0,1].set_title('loan status by Gender')

# Status by Education
sns.countplot(
    data=df,
    x='Education', 
    palette = {'Y':"steelblue",'N':"salmon"}, 
    edgecolor = 'k', 
    hue=TARGET_COL, 
    ax=axes[0,2], 
    linewidth = 0.5)
axes[0,2].set_title('loan status by Education')

# Hist plot by application income
axes[1,0].hist(
    df['ApplicantIncome'],
    bins=30,
    color='steelblue',
    edgecolor="k",
    linewidth=0.4)
axes[1,0].set_title('Applicant Income Distribution')
axes[1,0].set_xlabel("Income")

# Hist plot by Loan Amount
axes[1,1].hist(
    df['LoanAmount'],
    bins=30,
    color='magenta',
    edgecolor="k",
    linewidth=0.4)
axes[1,1].set_title('Loan Amount Distribution')
axes[1,1].set_xlabel("Loan Amount")

sns.countplot(
    data=df,
    x='Credit_History',
    hue = TARGET_COL,
    palette = {'Y':"steelblue",'N':"salmon"},
    ax=axes[1,2],
    edgecolor = 'k',
    linewidth=0.6
    )
axes[1,2].set_title('Loan Status by Credit History')
axes[1,2].set_xlabel("Credit History(1=Good, 0=Bad/Unknown)")


plt.tight_layout()
plt.savefig(
    "loan_plots.png",
    dpi =200,
    bbox_inches = 'tight')
#plt.show()

# Data Preprocessing
df_clean = df.copy()

for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col].fillna(
        df_clean[col].mode()[0],
        inplace=True)

for col in df_clean.select_dtypes(include='number').columns:
    df_clean[col].fillna(
        df_clean[col].median(),
        inplace=True)
    
# Encode Categorical columns

le = LabelEncoder()

cat_col = df_clean.select_dtypes(
    include=['object']).columns.tolist()
    
label_mapping ={}

for col in cat_col:
    df_clean[col] = le.fit_transform(df_clean[col])
    label_mapping[col] = dict(zip(le.classes_,le.transform(le.classes_)))

print("Label Encoding :")

for col,mapping in label_mapping.items():
    print(f"{col}: {mapping}")
    

# Feature Target Split

print(df_clean.head())

X = df_clean.drop(columns = [TARGET_COL])
y = df_clean[TARGET_COL]

print(f"Feature Columns: {list(X.columns)}")
print(f"Target Column: {TARGET_COL}")

print("\n"+"___"*45)

plt.figure(figsize=(18,10))
sns.heatmap(
    df_clean.corr(),
    annot=True,
    cmap="coolwarm",
    fmt =".2f",
    linewidth=2)

plt.title(
    "Correlation Heatmap",
    fontsize = 16,
    pad = 12)

plt.xticks(
    rotation=45,
    ha = 'right')

plt.yticks(rotation=45)
plt.tight_layout()

print("\n Correlation Matrix image saved to: 'loan_corr_heatmap.png'")

plt.savefig(
    "loan_heatmap.png",
    dpi=200,
    bbox_inches='tight')

plt.close()

# Split the Data into Training and Testing

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42,
    stratify = y
    )
    
print(f"Training Model samples : {len(X_train)}")
print(f"Testing Model samples : {len(X_test)}")

baseline = DecisionTreeClassifier(
    random_state = 42
    )

baseline.fit(X_train,y_train)
y_pred_base = baseline.predict(X_test)

print(f"Baseline Accuracy : {accuracy_score(y_test,y_pred_base):.4f}")

# HYPER PARAMETER TUNING

param_grid = {
    'criterion':['gini', 'entropy'],
    'splitter':['best', 'random'],
    'max_depth': [None,2,3,4,5,8,10,15],
    'min_samples_split':[2,5,10],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator = DecisionTreeClassifier(
        random_state=42
        ),
    param_grid = param_grid,
    scoring = 'accuracy',
    cv = 5,
    verbose = 2,
    n_jobs = -1
)

grid_search.fit(X_train,y_train)

for k,v in grid_search.best_params_.items():
    print(f"{k}: {v}")

y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_proba)
print(f"Accuracy : {acc:.4f}")
print(f"ROC - AUC : {roc_auc:.4f}")



# Model Evaluation

