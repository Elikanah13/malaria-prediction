# ================================
# MALARIA PREDICTION MODEL
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("Final_Malaria_Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ================================
# 3. Data Cleaning
# ================================

# Remove unnecessary columns
df = df.drop(columns=['ID','Notes','Disease_Cases','Avg_Income'], errors='ignore')

# Drop rows where target is missing
df = df.dropna(subset=['High_Risk_Binary'])

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ================================
# 4. Feature Selection
# ================================

X = df.drop(columns=['High_Risk_Binary'])
y = df['High_Risk_Binary']

categorical_cols = ['Region','County','Month']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ================================
# 5. Preprocessing Pipeline
# ================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ================================
# 6. Train Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 7. Models
# ================================

models = {

"Logistic Regression": Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
]),

"Random Forest": Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
]),

"XGBoost": Pipeline([
    ('prep', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    ))
])

}

# ================================
# 8. Train Models
# ================================

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = [acc, prec, rec, f1]

    print("\n====================")
    print(name)
    print("====================")
    print(classification_report(y_test, preds))

# ================================
# 9. Compare Models
# ================================

results_df = pd.DataFrame(results,
                          index=['Accuracy','Precision','Recall','F1']).T

print("\nModel Comparison")
print(results_df)

results_df.plot(kind='bar', figsize=(10,5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.show()

# ================================
# 10. Best Model
# ================================

best_model_name = results_df['Accuracy'].idxmax()
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# ================================
# 11. Confusion Matrix
# ================================

preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 12. Feature Importance (RF/XGB)
# ================================

if best_model_name != "Logistic Regression":

    model = best_model.named_steps['model']

    importances = model.feature_importances_

    plt.figure(figsize=(8,5))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.show()
