"""
Malaria Infection Prediction using Machine Learning
Group 3 — Meru University of Science and Technology
Models: Logistic Regression | Random Forest | Gradient Boosting (XGBoost-equivalent)
"""
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
 
# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("="*60)
print("  MALARIA INFECTION PREDICTION — ML PIPELINE")
print("="*60)
 
df = pd.read_csv('/mnt/user-data/uploads/Final_Malaria_Dataset.csv')
print(f"\n[DATA] Loaded {df.shape[0]} records, {df.shape[1]} columns")
print(f"[DATA] Columns: {list(df.columns)}")
print(f"\n[DATA] Class distribution (High_Risk_Binary):\n{df['High_Risk_Binary'].value_counts()}")
print(f"\n[DATA] Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
 
# ─────────────────────────────────────────────
# 2. PRE-PROCESSING & CLEANING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: DATA PRE-PROCESSING & CLEANING")
print("="*60)
 
# Drop mostly-empty columns (ID, Health_Facilities, Avg_Income, Disease_Cases, Notes)
drop_cols = ['ID', 'Health_Facilities', 'Avg_Income', 'Disease_Cases', 'Notes']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
 
# Remove duplicate records
before = len(df)
df.drop_duplicates(inplace=True)
print(f"[CLEAN] Removed {before - len(df)} duplicate rows")
 
# Standardise Region names
df['Region'] = df['Region'].str.strip().str.title()
df['County'] = df['County'].str.strip().str.title()
 
# Impute missing numerical values with median
num_cols = df.select_dtypes(include=np.number).columns.tolist()
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])
print(f"[CLEAN] Missing values imputed using median strategy")
 
# Outlier capping (IQR method) on key numeric features
cap_cols = ['Rainfall_mm', 'Temperature_C', 'Humidity_percent',
            'Malaria_Cases', 'Lag_1_Month_Cases', 'Incidence_per_100k']
for col in cap_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[col] = df[col].clip(lower, upper)
print(f"[CLEAN] Outliers capped using IQR method on: {cap_cols}")
 
# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & SELECTION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: FEATURE ENGINEERING & SELECTION")
print("="*60)
 
# Season from month
def get_season(m):
    if m in [3, 4, 5]:   return 'Long_Rains'
    elif m in [6, 7, 8]: return 'Dry'
    elif m in [9,10,11]: return 'Short_Rains'
    else:                return 'Cool_Dry'
 
df['Season'] = df['Month'].apply(get_season)
 
# Cases per capita
df['Cases_Per_Capita'] = df['Malaria_Cases'] / df['Population'] * 100000
 
# Encode categorical features
le_region  = LabelEncoder()
le_county  = LabelEncoder()
le_season  = LabelEncoder()
df['Region_enc'] = le_region.fit_transform(df['Region'])
df['County_enc'] = le_county.fit_transform(df['County'])
df['Season_enc'] = le_season.fit_transform(df['Season'])
 
# Feature set
FEATURES = ['Rainfall_mm', 'Temperature_C', 'Humidity_percent',
            'Lag_1_Month_Cases', 'Incidence_per_100k', 'Month',
            'Population', 'Malaria_Cases', 'Cases_Per_Capita',
            'Region_enc', 'County_enc', 'Season_enc']
 
TARGET = 'High_Risk_Binary'
 
X = df[FEATURES]
y = df[TARGET].astype(int)
 
print(f"[FEATURES] Using {len(FEATURES)} features: {FEATURES}")
print(f"[TARGET]   {TARGET}  |  0 = Low Risk, 1 = High Risk")
print(f"[SPLIT]    Class balance — Low: {(y==0).sum()}, High: {(y==1).sum()}")
 
# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
print(f"\n[SPLIT]  Train: {len(X_train)} | Test: {len(X_test)}")
 
# Scale for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
 
# ─────────────────────────────────────────────
# 5. MODEL TRAINING WITH HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: MODEL DEVELOPMENT & HYPERPARAMETER TUNING")
print("="*60)
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
# --- Logistic Regression ---
print("\n[MODEL 1] Logistic Regression — Grid Search...")
lr_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
lr_gs = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                     lr_params, cv=cv, scoring='f1', n_jobs=-1)
lr_gs.fit(X_train_sc, y_train)
lr_best = lr_gs.best_estimator_
print(f"  Best params: {lr_gs.best_params_}")
 
# --- Random Forest ---
print("\n[MODEL 2] Random Forest — Grid Search...")
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None],
             'min_samples_split': [2, 5]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                     rf_params, cv=cv, scoring='f1', n_jobs=-1)
rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_
print(f"  Best params: {rf_gs.best_params_}")
 
# --- Gradient Boosting (XGBoost-equivalent) ---
print("\n[MODEL 3] Gradient Boosting (XGBoost-equivalent) — Grid Search...")
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
             'max_depth': [3, 5]}
gb_gs = GridSearchCV(GradientBoostingClassifier(random_state=42),
                     gb_params, cv=cv, scoring='f1', n_jobs=-1)
gb_gs.fit(X_train, y_train)
gb_best = gb_gs.best_estimator_
print(f"  Best params: {gb_gs.best_params_}")
 
# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: PERFORMANCE EVALUATION")
print("="*60)
 
models = {
    'Logistic Regression': (lr_best, X_test_sc, X_train_sc),
    'Random Forest':       (rf_best, X_test, X_train),
    'Gradient Boosting':   (gb_best, X_test, X_train),
}
 
results = {}
for name, (model, Xte, Xtr) in models.items():
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1 Score':  f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_prob),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'cm':        confusion_matrix(y_test, y_pred),
    }
    print(f"\n  ─── {name} ───")
    print(f"  Accuracy : {results[name]['Accuracy']:.4f}")
    print(f"  Precision: {results[name]['Precision']:.4f}")
    print(f"  Recall   : {results[name]['Recall']:.4f}")
    print(f"  F1 Score : {results[name]['F1 Score']:.4f}")
    print(f"  ROC-AUC  : {results[name]['ROC-AUC']:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Low Risk','High Risk'])}")
 
# Best model
best_name = max(results, key=lambda k: results[k]['F1 Score'])
print(f"\n[WINNER] Best model by F1 Score: *** {best_name} *** "
      f"(F1={results[best_name]['F1 Score']:.4f})")
 
# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
print("\n[PLOTS] Generating visualisations...")
 
palette = {'Logistic Regression': '#3B82F6',
           'Random Forest':       '#10B981',
           'Gradient Boosting':   '#F59E0B'}
 
fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4)
fig.patch.set_facecolor('#F8FAFC')
 
# ── Title ──
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.65, 'Malaria Infection Prediction — ML Model Comparison',
              ha='center', va='center', fontsize=18, fontweight='bold', color='#1E293B')
ax_title.text(0.5, 0.2, 'Group 3 | Meru University of Science and Technology | Data Science',
              ha='center', va='center', fontsize=11, color='#64748B')
 
# ── (a) Metric comparison bar chart ──
ax1 = fig.add_subplot(gs[1, :])
metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x = np.arange(len(metric_cols))
width = 0.25
for i, (name, color) in enumerate(palette.items()):
    vals = [results[name][m] for m in metric_cols]
    bars = ax1.bar(x + i*width, vals, width, label=name, color=color, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color='#374151')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metric_cols, fontsize=11)
ax1.set_ylim(0, 1.12)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Model Performance Comparison Across All Metrics', fontsize=13, fontweight='bold', color='#1E293B')
ax1.legend(fontsize=10, loc='lower right')
ax1.set_facecolor('#F1F5F9')
ax1.spines[['top','right']].set_visible(False)
ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
 
# ── (b) Confusion matrices ──
for col_i, (name, color) in enumerate(palette.items()):
    ax = fig.add_subplot(gs[2, col_i])
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'],
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(f'Confusion Matrix\n{name}', fontsize=10, fontweight='bold', color='#1E293B')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
 
# ── (c) ROC curves ──
ax_roc = fig.add_subplot(gs[3, :2])
ax_roc.set_facecolor('#F1F5F9')
for name, color in palette.items():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['ROC-AUC']
    ax_roc.plot(fpr, tpr, color=color, lw=2.2, label=f'{name} (AUC={auc:.3f})')
ax_roc.plot([0,1],[0,1],'k--', lw=1, alpha=0.4, label='Random Classifier')
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('ROC Curves', fontsize=13, fontweight='bold', color='#1E293B')
ax_roc.legend(fontsize=10)
ax_roc.spines[['top','right']].set_visible(False)
 
# ── (d) Feature importance (RF) ──
ax_fi = fig.add_subplot(gs[3, 2])
fi = pd.Series(rf_best.feature_importances_, index=FEATURES).sort_values(ascending=True)
bars = ax_fi.barh(fi.index, fi.values, color='#10B981', alpha=0.8, edgecolor='white')
ax_fi.set_title('Feature Importance\n(Random Forest)', fontsize=11, fontweight='bold', color='#1E293B')
ax_fi.set_xlabel('Importance', fontsize=9)
ax_fi.set_facecolor('#F1F5F9')
ax_fi.spines[['top','right']].set_visible(False)
 
plt.savefig('/home/claude/malaria_model_results.png', dpi=160, bbox_inches='tight',
            facecolor='#F8FAFC')
plt.close()
print("[PLOTS] Saved → malaria_model_results.png")
 
# ─────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────
summary = pd.DataFrame({name: {m: round(v,4) for m,v in metrics.items()
                                if m not in ('y_pred','y_prob','cm')}
                        for name, metrics in results.items()}).T
summary.to_csv('/home/claude/model_summary.csv')
print("[OUTPUT] model_summary.csv saved")
print("\n" + summary.to_string())
print("\n" + "="*60)
print(f"  PIPELINE COMPLETE.  Best model: {best_name}")
print("="*60)
