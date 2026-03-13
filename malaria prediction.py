# ================================================================
#  MALARIA HIGH-RISK PREDICTOR  |  Full ML Pipeline
#  Target : High_Risk_Binary  (1 = High Risk, 0 = Low Risk)
#  Data   : 1500 Kenya county-month records  (2022 – 2026)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import (accuracy_score, roc_auc_score, f1_score,
                                      precision_score, recall_score,
                                      confusion_matrix, classification_report,
                                      roc_curve)
import joblib

SEP  = "=" * 68
DASH = "─" * 68

PAL  = ["#00d4ff", "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]
BG   = "#0f1523"
GRID = "#1e2d45"
TXT  = "#cbd5e1"

def _ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    if title:
        ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    return ax

# ── STEP 1: LOAD ─────────────────────────────────────────────────
print(SEP)
print("  STEP 1 › LOAD DATASET")
print(SEP)

df = pd.read_csv("Final_Malaria_Dataset.csv")
print(f"  Rows × Cols  : {df.shape}")
print(f"  Year range   : {df['Year'].min()} – {df['Year'].max()}")
print(f"  Regions      : {df['Region'].unique().tolist()}")
vc = df["High_Risk_Binary"].value_counts()
print(f"  Target split : High={vc[1]}  Low={vc[0]}  ({vc[1]/len(df)*100:.1f}% / {vc[0]/len(df)*100:.1f}%)")
print(f"\n{df.head(3).to_string()}\n")

# ── STEP 2: CLEAN ────────────────────────────────────────────────
print(SEP)
print("  STEP 2 › DATA CLEANING")
print(SEP)

print("  Missing values (before cleaning):")
for col, n in df.isnull().sum().items():
    pct = n / len(df) * 100
    print(f"    {col:<22}  {n:>5} ({pct:5.1f}%)  {'█'*int(pct/5)}")

# drop columns ≥ 40% null (ID, Health_Facilities, Avg_Income, Disease_Cases, Notes)
thresh    = 0.40
drop_cols = [c for c in df.columns if df[c].isnull().mean() >= thresh]
df.drop(columns=drop_cols, inplace=True)
print(f"\n  Dropped {len(drop_cols)} high-null cols: {drop_cols}")

n_before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed {n_before - len(df)} duplicate rows")

# impute any residual nulls
for col in df.select_dtypes("number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes("object").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"  Remaining nulls : {df.isnull().sum().sum()}")
print(f"  Clean shape     : {df.shape}")

# ── STEP 3: FEATURE ENG + SELECTION ──────────────────────────────
print(f"\n{SEP}")
print("  STEP 3 › FEATURE ENGINEERING & SELECTION")
print(SEP)

le_region = LabelEncoder()
le_county = LabelEncoder()
df["Region_enc"]     = le_region.fit_transform(df["Region"])
df["County_enc"]     = le_county.fit_transform(df["County"])
df["Cases_per_Pop"]  = df["Malaria_Cases"] / df["Population"] * 1e5
df["Lag_Change"]     = df["Malaria_Cases"] - df["Lag_1_Month_Cases"]
df["Lag_Change_Pct"] = df["Lag_Change"] / (df["Lag_1_Month_Cases"] + 1)
df["Rain_x_Temp"]    = df["Rainfall_mm"] * df["Temperature_C"]
df["Humidity_x_Rain"]= df["Humidity_percent"] * df["Rainfall_mm"]
df["Season_enc"]     = LabelEncoder().fit_transform(
    df["Month"].map(lambda m: "LongRain" if m in [3,4,5] else
                              "ShortRain" if m in [10,11,12] else "Dry"))

TARGET       = "High_Risk_Binary"
EXCLUDE      = ["Region", "County", TARGET]
feature_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feature_cols]
y = df[TARGET]

print(f"  Total engineered features: {len(feature_cols)}")
for f in feature_cols:
    print(f"    • {f}")

K        = 10
selector = SelectKBest(f_classif, k=K)
selector.fit(X, y)
scores_s = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
TOP_FEATS= scores_s.head(K).index.tolist()
X_sel    = X[TOP_FEATS]

print(f"\n  Top {K} features (ANOVA F-score):")
for feat, sc in scores_s.head(K).items():
    print(f"    {feat:<28}  F={sc:>8.2f}  {'█'*min(int(sc/80),28)}")

# ── STEP 4: SPLIT ────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 › TRAIN / TEST SPLIT")
print(SEP)

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.20, random_state=42, stratify=y)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

print(f"  Train : {len(X_train)} rows  (High={y_train.sum()} | Low={(y_train==0).sum()})")
print(f"  Test  : {len(X_test)}  rows  (High={y_test.sum()}  | Low={(y_test==0).sum()})")
print(f"  Split : 80/20 stratified | Scaling: StandardScaler")

# ── STEP 5: CHOOSE ALGO ──────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 5 › CHOOSE ALGORITHM")
print(SEP)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   min_samples_leaf=2, random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                       max_depth=5, random_state=42),
    "SVM (RBF)"          : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}
for name in MODELS:
    print(f"  ✔  {name}")

# ── STEP 6: TRAIN ────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 6 › TRAIN MODELS  (5-Fold Stratified CV)")
print(SEP)

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in MODELS.items():
    model.fit(X_tr_sc, y_train)
    cv_auc = cross_val_score(model, X_tr_sc, y_train, cv=skf, scoring="roc_auc")
    cv_acc = cross_val_score(model, X_tr_sc, y_train, cv=skf, scoring="accuracy")
    results[name] = {"model": model,
                     "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                     "cv_acc_mean": cv_acc.mean()}
    print(f"  {name:<25}  CV-AUC {cv_auc.mean():.4f} ± {cv_auc.std():.4f}  "
          f"| CV-Acc {cv_acc.mean():.4f}")

# ── STEP 7: EVALUATE ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 7 › EVALUATE ON HOLD-OUT TEST SET")
print(SEP)

for name, info in results.items():
    m = info["model"]
    yp = m.predict(X_te_sc);  yproba = m.predict_proba(X_te_sc)[:, 1]
    info.update({"y_pred": yp, "y_proba": yproba,
                 "acc": accuracy_score(y_test, yp),
                 "auc": roc_auc_score(y_test, yproba),
                 "f1" : f1_score(y_test, yp),
                 "prec": precision_score(y_test, yp),
                 "rec" : recall_score(y_test, yp),
                 "cm"  : confusion_matrix(y_test, yp)})

hdr = f"{'Model':<25} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}"
print(f"\n  {hdr}")
print(f"  {DASH}")
for name, info in results.items():
    star = " ★" if name == max(results, key=lambda k: results[k]["auc"]) else ""
    print(f"  {name:<25} {info['acc']:>7.4f} {info['auc']:>7.4f} "
          f"{info['f1']:>7.4f} {info['prec']:>7.4f} {info['rec']:>7.4f}{star}")

for name, info in results.items():
    print(f"\n  {DASH}\n  {name}")
    print(classification_report(y_test, info["y_pred"],
                                 target_names=["Low Risk","High Risk"], digits=4))

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n  ★  Best › {best_name}  AUC={best['auc']:.4f}  F1={best['f1']:.4f}")

# ── STEP 8: DEPLOY ───────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 8 › DEPLOY  –  SAVE ARTIFACTS")
print(SEP)

joblib.dump(best["model"],  "malaria_model.pkl")
joblib.dump(scaler,         "malaria_scaler.pkl")
joblib.dump(le_region,      "malaria_le_region.pkl")
joblib.dump(le_county,      "malaria_le_county.pkl")
pd.Series(TOP_FEATS, name="feature").to_csv("malaria_features.csv", index=False)
for f in ["malaria_model.pkl","malaria_scaler.pkl",
          "malaria_le_region.pkl","malaria_le_county.pkl","malaria_features.csv"]:
    print(f"  ✔  Saved  {f}")

def predict_risk(input_dict: dict) -> dict:
    """Predict malaria risk for one new observation.
    Keys must match features in malaria_features.csv."""
    _m  = joblib.load("malaria_model.pkl")
    _sc = joblib.load("malaria_scaler.pkl")
    _ft = pd.read_csv("malaria_features.csv")["feature"].tolist()
    sc  = _sc.transform(pd.DataFrame([input_dict])[_ft])
    p   = _m.predict(sc)[0];  pr = _m.predict_proba(sc)[0]
    return {"risk_level": "🔴 HIGH RISK" if p==1 else "🟢 LOW RISK",
            "confidence": f"{max(pr)*100:.1f}%",
            "prob_high": round(float(pr[1]),4), "prob_low": round(float(pr[0]),4)}

demo = predict_risk(X_test.iloc[0].to_dict())
print(f"\n  Demo (test row 0) ► {demo}")

# ── PLOTS ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  GENERATING PLOTS  ›  malaria_pipeline_results.png")
print(SEP)

fig = plt.figure(figsize=(22, 16), facecolor="#080c14")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)
fig.suptitle("Malaria High-Risk Predictor  –  Full ML Pipeline Results",
             color="white", fontsize=15, fontweight="bold", y=0.98)
SHORT = ["LR","RF","GBM","SVM"]

# 1 target dist
ax = _ax(fig.add_subplot(gs[0,0]), "Target Distribution")
cnt = y.value_counts()
bs  = ax.bar(["Low Risk","High Risk"],[cnt[0],cnt[1]], color=["#34d399","#fb7185"], edgecolor=GRID, width=0.5)
for b,v in zip(bs,[cnt[0],cnt[1]]):
    ax.text(b.get_x()+b.get_width()/2, v+5, str(v), ha="center", color=TXT, fontweight="bold", fontsize=10)
ax.set_ylabel("Count", color=TXT)

# 2 ANOVA scores
ax = _ax(fig.add_subplot(gs[0,1:3]), "Feature Selection  –  ANOVA F-Score (Top 10)")
top = scores_s.head(10).sort_values()
ax.barh(top.index, top.values, color=[PAL[i%len(PAL)] for i in range(len(top))], edgecolor=GRID)
for i,(v,l) in enumerate(zip(top.values,top.index)):
    ax.text(v+10, i, f"{v:.0f}", va="center", color=TXT, fontsize=8)
ax.set_xlabel("F-Score", color=TXT)

# 3 monthly avg
ax = _ax(fig.add_subplot(gs[0,3]), "Avg Cases by Month")
mon = df.groupby("Month")["Malaria_Cases"].mean()
ax.plot(mon.index, mon.values, color=PAL[0], lw=2.5, marker="o", ms=4)
ax.fill_between(mon.index, mon.values, alpha=0.15, color=PAL[0])
ax.set_xlabel("Month", color=TXT); ax.set_ylabel("Avg Cases", color=TXT)
ax.set_xticks(range(1,13))

# 4 CV AUC
ax = _ax(fig.add_subplot(gs[1,0]), "5-Fold CV  AUC-ROC")
cvm = [results[n]["cv_auc_mean"] for n in MODELS]
cvs = [results[n]["cv_auc_std"]  for n in MODELS]
bs  = ax.bar(SHORT, cvm, yerr=cvs, color=PAL[:4], capsize=5, edgecolor=GRID)
ax.set_ylim(0,1.12); ax.set_ylabel("AUC", color=TXT)
for b,v in zip(bs,cvm):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center", color=TXT, fontsize=8, fontweight="bold")

# 5 test metrics
ax = _ax(fig.add_subplot(gs[1,1]), "Test Set Metrics  –  All Models")
mets=["acc","auc","f1","prec","rec"]; labs=["Acc","AUC","F1","Prec","Rec"]
xb = np.arange(len(mets)); w=0.18
for i,(name,short) in enumerate(zip(MODELS,SHORT)):
    ax.bar(xb+i*w, [results[name][m] for m in mets], w, label=short, color=PAL[i], edgecolor=GRID, alpha=0.88)
ax.set_xticks(xb+w*1.5); ax.set_xticklabels(labs, color=TXT, fontsize=8)
ax.set_ylim(0,1.15); ax.set_ylabel("Score", color=TXT)
ax.legend(labelcolor=TXT, facecolor=BG, fontsize=8, ncol=2, framealpha=0.4)

# 6 ROC
ax = _ax(fig.add_subplot(gs[1,2]), "ROC Curves")
for i,(name,short) in enumerate(zip(MODELS,SHORT)):
    fpr,tpr,_ = roc_curve(y_test, results[name]["y_proba"])
    ax.plot(fpr, tpr, color=PAL[i], lw=2, label=f"{short}  AUC={results[name]['auc']:.3f}")
ax.plot([0,1],[0,1],"--",color=GRID,lw=1)
ax.set_xlabel("FPR",color=TXT); ax.set_ylabel("TPR",color=TXT)
ax.legend(labelcolor=TXT, facecolor=BG, fontsize=8, framealpha=0.4)

# 7 confusion matrix
ax = _ax(fig.add_subplot(gs[1,3]), f"Confusion Matrix\n({best_name})")
cm = best["cm"]
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Low","High"], color=TXT)
ax.set_yticklabels(["Low","High"], color=TXT)
ax.set_xlabel("Predicted",color=TXT); ax.set_ylabel("Actual",color=TXT)
for (r,c),val in zip([(0,0),(0,1),(1,0),(1,1)], cm.ravel()):
    ax.text(c, r, val, ha="center", va="center", color="white", fontsize=18, fontweight="bold")

# 8 incidence distribution
ax = _ax(fig.add_subplot(gs[2,0:2]), "Incidence per 100k  –  Risk Groups")
ax.hist(df[df[TARGET]==0]["Incidence_per_100k"], bins=45, color="#34d399", alpha=0.72, label="Low Risk",  edgecolor=GRID)
ax.hist(df[df[TARGET]==1]["Incidence_per_100k"], bins=45, color="#fb7185", alpha=0.72, label="High Risk", edgecolor=GRID)
ax.set_xlabel("Incidence per 100k",color=TXT); ax.set_ylabel("Frequency",color=TXT)
ax.legend(labelcolor=TXT, facecolor=BG, fontsize=9, framealpha=0.4)

# 9 cases by region
ax = _ax(fig.add_subplot(gs[2,2]), "Avg Cases by Region")
rg = df.groupby("Region")["Malaria_Cases"].mean().sort_values()
ax.barh([r.replace(" Region","") for r in rg.index], rg.values,
        color=PAL[:len(rg)], edgecolor=GRID)
ax.set_xlabel("Avg Cases",color=TXT)
for i,v in enumerate(rg.values):
    ax.text(v+10, i, f"{v:.0f}", va="center", color=TXT, fontsize=9)

# 10 pipeline summary
ax = _ax(fig.add_subplot(gs[2,3]), "Pipeline Summary")
ax.axis("off")
steps = [
    ("①","Load Dataset",      "1500 rows × 17 cols"),
    ("②","Data Cleaning",     "Dropped 5 high-null cols"),
    ("③","Feature Eng.",      "+7 derived features"),
    ("④","Feature Selection", f"Top {K} via ANOVA F-score"),
    ("⑤","Train/Test Split",  "80/20  stratified"),
    ("⑥","Train Models",      "LR · RF · GBM · SVM"),
    ("⑦","Best Model",        best_name),
    ("⑧","Metrics",           f"AUC={best['auc']:.4f}  F1={best['f1']:.4f}"),
    ("⑨","Deploy",            ".pkl artifacts saved ✓"),
]
for i,(num,title,detail) in enumerate(steps):
    yp = 0.97 - i*0.107
    ax.text(0.01, yp,       num,    color=PAL[i%len(PAL)], fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.14, yp,       title,  color=TXT, fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.14, yp-0.045, detail, color="#64748b", fontsize=7.5, style="italic", transform=ax.transAxes)

plt.savefig("malaria_pipeline_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✔  malaria_pipeline_results.png saved")
print(f"\n{SEP}")
print("  ✅  PIPELINE COMPLETE")
print(SEP)
