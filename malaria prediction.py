# ================================================================
#  MALARIA HIGH-RISK PREDICTOR — Complete ML Pipeline
#  Target : High_Risk_Binary  (1 = High Risk | 0 = Low Risk)
#  Data   : Kenya regions — 1500 records (2022–2026)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, joblib, os
warnings.filterwarnings("ignore")

from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection  import SelectKBest, f_classif
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.metrics            import (accuracy_score, f1_score, precision_score,
                                        recall_score, roc_auc_score, roc_curve,
                                        confusion_matrix, classification_report)

PAL   = ["#00d4ff", "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]
BG    = "#0f1523"
GRID  = "#1e2d45"
TEXT  = "#cbd5e1"
BLACK = "#080c14"

def _ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    if title: ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    return ax

def divider(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ================================================================
# STEP 1 — LOAD DATASET
# ================================================================
divider("STEP 1  LOAD DATASET")

df = pd.read_csv("Final_Malaria_Dataset.csv")
vc = df["High_Risk_Binary"].value_counts()
print(f"  Rows x Cols   : {df.shape}")
print(f"  Regions       : {sorted(df['Region'].unique())}")
print(f"  Counties      : {df['County'].nunique()} unique")
print(f"  Year range    : {df['Year'].min()} - {df['Year'].max()}")
print(f"  Low  Risk (0) : {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  High Risk (1) : {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")

# ================================================================
# STEP 2 — DATA CLEANING
# ================================================================
divider("STEP 2  DATA CLEANING")

miss_before = df.isnull().sum()
print("  Missing values per column (before):")
for col, n in miss_before[miss_before > 0].items():
    print(f"    {col:<22}: {n:>4}  ({n/len(df)*100:.0f}%)")

# Drop columns with >= 40% missing  (ID, Health_Facilities, Avg_Income, Disease_Cases, Notes)
drop_cols = [c for c in df.columns if df[c].isnull().mean() >= 0.40]
df.drop(columns=drop_cols, inplace=True)
print(f"\n  Dropped columns (>=40% null) : {drop_cols}")

# Remove duplicate rows
n_dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"  Duplicate rows removed       : {n_dup}")

# Impute any residual nulls
for col in df.select_dtypes(include="number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"  Remaining nulls              : {df.isnull().sum().sum()}")
print(f"  Shape after cleaning         : {df.shape}")

# ================================================================
# STEP 3 — FEATURE ENGINEERING & SELECTION
# ================================================================
divider("STEP 3  FEATURE ENGINEERING & SELECTION")

# Encode categoricals
le_region = LabelEncoder()
le_county = LabelEncoder()
df["Region_enc"] = le_region.fit_transform(df["Region"])
df["County_enc"] = le_county.fit_transform(df["County"])

# Seasonal buckets (Kenya: long rains Mar-May, short rains Oct-Dec)
df["Season"] = df["Month"].map(
    {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3,12:3})

# Derived numeric features
df["Cases_per_Pop"]   = df["Malaria_Cases"] / df["Population"] * 1e5
df["Lag_Change"]      = df["Malaria_Cases"] - df["Lag_1_Month_Cases"]
df["Lag_Ratio"]       = (df["Malaria_Cases"] /
                          df["Lag_1_Month_Cases"].replace(0, np.nan)).fillna(1)
df["Rain_x_Humidity"] = df["Rainfall_mm"]   * df["Humidity_percent"]
df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity_percent"]
df["Rain_x_Temp"]     = df["Rainfall_mm"]   * df["Temperature_C"]

new_feats = ["Season","Cases_per_Pop","Lag_Change","Lag_Ratio",
             "Rain_x_Humidity","Temp_x_Humidity","Rain_x_Temp"]
print(f"  New features created : {new_feats}")

TARGET    = "High_Risk_Binary"
EXCLUDE   = ["Region", "County", TARGET]
feat_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feat_cols]
y = df[TARGET]
print(f"\n  Total candidate features : {len(feat_cols)}")

# SelectKBest (ANOVA F-score)
K        = 12
selector = SelectKBest(f_classif, k=K)
selector.fit(X, y)
f_scores = pd.Series(selector.scores_, index=feat_cols).sort_values(ascending=False)
sel_feats = f_scores.head(K).index.tolist()
X_sel     = X[sel_feats]

print(f"\n  Top {K} features by ANOVA F-score:")
for feat, score in f_scores.head(K).items():
    print(f"    * {feat:<28}  F = {score:>10.2f}")

# ================================================================
# STEP 4 — TRAIN / TEST SPLIT
# ================================================================
divider("STEP 4  TRAIN / TEST SPLIT  (80/20 stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.20, random_state=42, stratify=y)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Training : {len(X_train):>5}  (High={y_train.sum()} | Low={(y_train==0).sum()})")
print(f"  Testing  : {len(X_test):>5}  (High={y_test.sum()} | Low={(y_test==0).sum()})")

# ================================================================
# STEP 5 & 6 — CHOOSE & TRAIN MODELS
# ================================================================
divider("STEP 5 & 6  CHOOSE ALGORITHMS & TRAIN")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "Logistic Regression" : LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=300, max_depth=12,
                                                    min_samples_leaf=2, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=5, subsample=0.8,
                                                        random_state=42),
    "SVM (RBF)"           : SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
}

results = {}
print(f"\n  {'Model':<25}  CV-AUC (mean)   CV-AUC (std)")
print(f"  {'-'*55}")
for name, model in MODELS.items():
    model.fit(X_train_sc, y_train)
    cv = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="roc_auc")
    results[name] = dict(model=model, cv_mean=cv.mean(), cv_std=cv.std())
    print(f"  {name:<25}  {cv.mean():.4f}          {cv.std():.4f}")

# ================================================================
# STEP 7 — EVALUATE
# ================================================================
divider("STEP 7  EVALUATE ON HELD-OUT TEST SET")

for name, info in results.items():
    m       = info["model"]
    y_pred  = m.predict(X_test_sc)
    y_proba = m.predict_proba(X_test_sc)[:, 1]
    info.update(dict(
        y_pred  = y_pred, y_proba = y_proba,
        acc     = accuracy_score(y_test, y_pred),
        auc     = roc_auc_score(y_test, y_proba),
        f1      = f1_score(y_test, y_pred),
        prec    = precision_score(y_test, y_pred),
        rec     = recall_score(y_test, y_pred),
        cm      = confusion_matrix(y_test, y_pred),
        report  = classification_report(y_test, y_pred,
                      target_names=["Low Risk","High Risk"], output_dict=True),
    ))

print(f"\n  {'Model':<25}  Accuracy    AUC       F1      Precision   Recall")
print(f"  {'-'*72}")
for name, info in results.items():
    print(f"  {name:<25}  {info['acc']:.4f}    {info['auc']:.4f}    "
          f"{info['f1']:.4f}    {info['prec']:.4f}    {info['rec']:.4f}")

best_name  = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\n  BEST MODEL : {best_name}")
print(f"  AUC={results[best_name]['auc']:.4f}  F1={results[best_name]['f1']:.4f}  "
      f"Accuracy={results[best_name]['acc']:.4f}")
print(f"\n  Detailed Report - {best_name}:")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=["Low Risk","High Risk"]))

# ================================================================
# STEP 8 — DEPLOY
# ================================================================
divider("STEP 8  DEPLOY - SAVE ARTIFACTS")

joblib.dump(best_model, "malaria_model.pkl")
joblib.dump(scaler,     "malaria_scaler.pkl")
joblib.dump(le_region,  "malaria_le_region.pkl")
joblib.dump(le_county,  "malaria_le_county.pkl")
pd.Series(sel_feats, name="feature").to_csv("malaria_features.csv", index=False)

for f in ["malaria_model.pkl","malaria_scaler.pkl",
          "malaria_le_region.pkl","malaria_le_county.pkl","malaria_features.csv"]:
    print(f"  Saved: {f:<40} ({os.path.getsize(f)/1024:.1f} KB)")


def predict_risk(input_dict: dict) -> dict:
    """
    Predict malaria High-Risk status for a new observation.

    Parameters
    ----------
    input_dict : dict
        Must contain keys matching malaria_features.csv (pre-encoded numerics).

    Returns
    -------
    dict : risk_level, confidence, prob_high, prob_low

    Example
    -------
    >>> predict_risk({
    ...     "Incidence_per_100k": 145.0,
    ...     "Cases_per_Pop": 145.0,
    ...     "Malaria_Cases": 1200,
    ...     "Lag_1_Month_Cases": 1100,
    ...     "Population": 900000,
    ...     "Lag_Change": 100,
    ...     "Lag_Ratio": 1.09,
    ...     "Rainfall_mm": 180.0,
    ...     "Rain_x_Humidity": 13680.0,
    ...     "Humidity_percent": 76.0,
    ...     "Temp_x_Humidity": 1880.0,
    ...     "Rain_x_Temp": 4320.0,
    ... })
    """
    _model  = joblib.load("malaria_model.pkl")
    _scaler = joblib.load("malaria_scaler.pkl")
    _feats  = pd.read_csv("malaria_features.csv").squeeze().tolist()
    row     = pd.DataFrame([input_dict])[_feats]
    row_sc  = _scaler.transform(row)
    pred    = _model.predict(row_sc)[0]
    proba   = _model.predict_proba(row_sc)[0]
    return {
        "risk_level" : "HIGH RISK" if pred == 1 else "LOW RISK",
        "confidence" : f"{max(proba)*100:.1f}%",
        "prob_high"  : round(float(proba[1]), 4),
        "prob_low"   : round(float(proba[0]), 4),
    }

# demo
demo = predict_risk(dict(zip(sel_feats, X_test.iloc[-1].values)))
print(f"\n  Demo prediction (last test row): {demo}")

# ================================================================
# VISUALISATIONS
# ================================================================
print("\n  Building pipeline_results.png ...")

fig = plt.figure(figsize=(22, 18), facecolor=BLACK)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

# 1. Target distribution
ax1 = _ax(fig.add_subplot(gs[0, 0]), "Target Distribution")
bars1 = ax1.bar(["Low Risk (0)", "High Risk (1)"],
                [vc[0], vc[1]], color=["#34d399","#fb7185"],
                edgecolor=GRID, width=0.5)
for b in bars1:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+8,
             str(int(b.get_height())), ha="center", color=TEXT,
             fontweight="bold", fontsize=11)
ax1.set_ylabel("Count", color=TEXT)

# 2. ANOVA F-scores
ax2 = _ax(fig.add_subplot(gs[0, 1:3]), "ANOVA F-Scores — Top Selected Features")
fs  = f_scores.head(K).sort_values()
ax2.barh(fs.index, fs.values,
         color=[PAL[i % len(PAL)] for i in range(len(fs))],
         edgecolor=GRID, height=0.65)
ax2.set_xlabel("F-Score", color=TEXT)
for i, (val, lbl) in enumerate(zip(fs.values, fs.index)):
    ax2.text(val + fs.max()*0.01, i, f"{val:.0f}",
             va="center", color=TEXT, fontsize=7.5)

# 3. Cases by region
ax3 = _ax(fig.add_subplot(gs[0, 3]), "Avg Cases by Region")
rg  = df.groupby("Region")["Malaria_Cases"].mean().sort_values()
ax3.barh([r.replace(" Region","") for r in rg.index],
         rg.values, color=PAL[:len(rg)], edgecolor=GRID)
ax3.set_xlabel("Avg Cases", color=TEXT)

# 4. CV AUC
ax4 = _ax(fig.add_subplot(gs[1, 0]), "5-Fold CV AUC by Model")
short = ["LR","RF","GBM","SVM"]
means = [results[n]["cv_mean"] for n in MODELS]
stds  = [results[n]["cv_std"]  for n in MODELS]
bars4 = ax4.bar(short, means, yerr=stds, color=PAL[:4],
                capsize=6, edgecolor=GRID, width=0.55)
ax4.set_ylim(0, 1.12)
ax4.set_ylabel("AUC", color=TEXT)
for b, v in zip(bars4, means):
    ax4.text(b.get_x()+b.get_width()/2, v+0.03,
             f"{v:.4f}", ha="center", color=TEXT, fontsize=8)

# 5. Test metrics
ax5 = _ax(fig.add_subplot(gs[1, 1]), "Test Metrics — All Models")
mets  = ["acc","auc","f1","prec","rec"]
xlbls = ["Acc","AUC","F1","Prec","Recall"]
x, w  = np.arange(len(mets)), 0.18
for i, (name, info) in enumerate(results.items()):
    ax5.bar(x + i*w, [info[m] for m in mets], w,
            label=short[i], color=PAL[i], edgecolor=GRID, alpha=0.88)
ax5.set_xticks(x + w*1.5)
ax5.set_xticklabels(xlbls, color=TEXT, fontsize=8)
ax5.set_ylim(0, 1.15)
ax5.legend(labelcolor=TEXT, facecolor=BG, fontsize=8, ncol=2, framealpha=0.4)

# 6. ROC curves
ax6 = _ax(fig.add_subplot(gs[1, 2]), "ROC Curves")
for i, (name, info) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, info["y_proba"])
    ax6.plot(fpr, tpr, color=PAL[i], lw=2,
             label=f"{short[i]}  AUC={info['auc']:.3f}")
ax6.plot([0,1],[0,1],"--",color=GRID,lw=1)
ax6.set_xlabel("False Positive Rate", color=TEXT)
ax6.set_ylabel("True Positive Rate", color=TEXT)
ax6.legend(labelcolor=TEXT, facecolor=BG, fontsize=8, framealpha=0.4)

# 7. Confusion matrix
ax7 = _ax(fig.add_subplot(gs[1, 3]), f"Confusion Matrix ({best_name[:10]}...)")
cm  = results[best_name]["cm"]
ax7.imshow(cm, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax7.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                 color="white", fontsize=18, fontweight="bold")
ax7.set_xticks([0,1]); ax7.set_yticks([0,1])
ax7.set_xticklabels(["Low","High"], color=TEXT)
ax7.set_yticklabels(["Low","High"], color=TEXT)
ax7.set_xlabel("Predicted", color=TEXT); ax7.set_ylabel("Actual", color=TEXT)

# 8. Incidence histograms
ax8 = _ax(fig.add_subplot(gs[2, 0:2]), "Incidence per 100k — Risk Groups")
low_i  = df[df[TARGET]==0]["Incidence_per_100k"]
high_i = df[df[TARGET]==1]["Incidence_per_100k"]
ax8.hist(low_i,  bins=45, color="#34d399", alpha=0.70,
         label=f"Low Risk  (n={len(low_i)})", edgecolor=GRID)
ax8.hist(high_i, bins=45, color="#fb7185", alpha=0.70,
         label=f"High Risk (n={len(high_i)})", edgecolor=GRID)
ax8.axvline(df["Incidence_per_100k"].median(),
            color="#fbbf24", linestyle="--", lw=1.5, label="Overall Median")
ax8.set_xlabel("Incidence per 100 000", color=TEXT)
ax8.set_ylabel("Frequency", color=TEXT)
ax8.legend(labelcolor=TEXT, facecolor=BG, fontsize=9, framealpha=0.4)

# 9. Monthly trend
ax9 = _ax(fig.add_subplot(gs[2, 2]), "Avg Malaria Cases by Month")
mo  = df.groupby("Month")["Malaria_Cases"].mean()
ax9.fill_between(mo.index, mo.values, alpha=0.18, color=PAL[0])
ax9.plot(mo.index, mo.values, color=PAL[0], lw=2.5,
         marker="o", markersize=5)
ax9.set_xlabel("Month", color=TEXT)
ax9.set_ylabel("Avg Cases", color=TEXT)
ax9.set_xticks(range(1, 13))
for xv in [3, 10]:
    ax9.axvspan(xv, xv+3, alpha=0.07, color="#34d399")

# 10. Pipeline card
ax10 = _ax(fig.add_subplot(gs[2, 3]), "Pipeline Summary")
ax10.axis("off")
info_best = results[best_name]
lines = [
    ("DATASET",       "1 500 rows  3 regions  21 counties"),
    ("CLEANING",      "Dropped 5 cols (>=40% null), 0 dupes"),
    ("ENGINEERING",   "7 new interaction features derived"),
    ("SELECTION",     f"Top {K} via ANOVA F-score"),
    ("SPLIT",         "80 / 20  stratified"),
    ("MODELS",        "LR  RF  GBM  SVM"),
    ("BEST MODEL",    best_name),
    ("  AUC",         f"{info_best['auc']:.4f}"),
    ("  F1 Score",    f"{info_best['f1']:.4f}"),
    ("  Accuracy",    f"{info_best['acc']:.4f}"),
    ("DEPLOY",        "4 .pkl + features.csv  saved"),
]
for i, (lbl, val) in enumerate(lines):
    yp = 0.97 - i * 0.088
    ax10.text(0.02, yp, lbl, color=PAL[i % len(PAL)], fontsize=8,
              fontweight="bold", transform=ax10.transAxes, fontfamily="monospace")
    ax10.text(0.42, yp, val, color=TEXT, fontsize=7.8, transform=ax10.transAxes)

fig.suptitle("Malaria High-Risk Predictor  -  Full ML Pipeline",
             color="white", fontsize=17, fontweight="bold", y=0.998)

plt.savefig("pipeline_results.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  pipeline_results.png saved")
print(f"\n{'='*65}")
print("  PIPELINE COMPLETE")
print(f"{'='*65}")
