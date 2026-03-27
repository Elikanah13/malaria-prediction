"""
=============================================================================
  MALARIA INFECTION FORECASTING PIPELINE
  Meru University of Science and Technology — Group 3, Data Science Project
  Regions: Nyanza | Rift Valley | Central Kenya  |  Period: 2022–2026
=============================================================================
  Models  : Random Forest | XGBoost | Logistic Regression (High-Risk classifier)
  Target  : Malaria_Cases (regression) & High_Risk_Binary (classification)
  Extras  : SARIMA time-series forecast for future trend projection
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

# Optional SARIMA (statsmodels) for time-series forecast
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    print("[WARNING] statsmodels not installed — SARIMA step will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "Final_Malaria_Dataset.csv"   # ← update path if needed

def load_data(path: str) -> pd.DataFrame:
    """Load dataset and do a quick sanity-check printout."""
    df = pd.read_csv(path)
    print("=" * 65)
    print(f"  Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 65)
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print()
    return df

df = load_data(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA PRE-PROCESSING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, impute, encode, and normalise the raw dataset."""

    df = df.copy()

    # --- 2a. Drop irrelevant / very-sparse columns
    drop_cols = ["ID", "Notes", "Disease_Cases"]   # >40 % missing or unused
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- 2b. Impute remaining numeric NaNs with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer  = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # --- 2c. Remove exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before - len(df)} duplicate rows.")

    # --- 2d. Outlier capping (IQR × 3) on Malaria_Cases
    for col in ["Malaria_Cases", "Rainfall_mm", "Incidence_per_100k"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)

    # --- 2e. Encode categorical: Region and County
    le_region = LabelEncoder()
    le_county = LabelEncoder()
    df["Region_enc"] = le_region.fit_transform(df["Region"])
    df["County_enc"] = le_county.fit_transform(df["County"])

    # Store encoder mappings for reference
    df.attrs["region_classes"] = list(le_region.classes_)
    df.attrs["county_classes"] = list(le_county.classes_)

    print("  Encoding done.")
    print(f"  Regions : {le_region.classes_.tolist()}")
    return df

df = preprocess(df)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and domain-driven features."""

    df = df.copy()

    # --- 3a. Date column for sorting / time-series work
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )

    # --- 3b. Cyclical encoding of Month (captures seasonality)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # --- 3c. Interaction features
    df["Rainfall_Temp"]     = df["Rainfall_mm"]    * df["Temperature_C"]
    df["Humidity_Temp"]     = df["Humidity_percent"] * df["Temperature_C"]
    df["Rainfall_Humidity"] = df["Rainfall_mm"]    * df["Humidity_percent"]

    # --- 3d. Lag features (already present: Lag_1_Month_Cases)
    #         Add Lag-2 using shift within (Region, County) groups
    df.sort_values(["Region", "County", "Year", "Month"], inplace=True)
    grp = df.groupby(["Region", "County"])
    df["Lag_2_Month_Cases"] = grp["Malaria_Cases"].shift(2)
    df["Lag_3_Month_Cases"] = grp["Malaria_Cases"].shift(3)
    df["Rolling_3M_Mean"]   = grp["Malaria_Cases"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # --- 3e. Population density proxy (cases per person)
    df["Cases_per_pop"] = df["Malaria_Cases"] / df["Population"]

    # --- 3f. Fill newly introduced NaNs with median
    new_lag_cols = ["Lag_2_Month_Cases", "Lag_3_Month_Cases", "Rolling_3M_Mean"]
    df[new_lag_cols] = df[new_lag_cols].fillna(df[new_lag_cols].median())

    print(f"  Feature engineering done. Total features: {df.shape[1]}")
    return df

df = engineer_features(df)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Columns to use as model input
FEATURE_COLS = [
    "Region_enc", "County_enc", "Year", "Month_sin", "Month_cos",
    "Population", "Rainfall_mm", "Temperature_C", "Humidity_percent",
    "Lag_1_Month_Cases", "Lag_2_Month_Cases", "Lag_3_Month_Cases",
    "Rolling_3M_Mean", "Rainfall_Temp", "Humidity_Temp",
    "Rainfall_Humidity", "Health_Facilities", "Avg_Income"
]

# Targets
TARGET_REG   = "Malaria_Cases"         # regression
TARGET_CLASS = "High_Risk_Binary"      # classification

X = df[FEATURE_COLS]
y_reg   = df[TARGET_REG]
y_class = df[TARGET_CLASS]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN / TEST SPLIT  (80 / 20, time-aware)
# ─────────────────────────────────────────────────────────────────────────────

# Time-sorted split keeps temporal order (no data leakage)
df_sorted = df.sort_values("Date").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.80)

train_df = df_sorted.iloc[:split_idx]
test_df  = df_sorted.iloc[split_idx:]

X_train = train_df[FEATURE_COLS]
X_test  = test_df[FEATURE_COLS]
y_train_reg   = train_df[TARGET_REG]
y_test_reg    = test_df[TARGET_REG]
y_train_class = train_df[TARGET_CLASS]
y_test_class  = test_df[TARGET_CLASS]

# Scale features for Logistic Regression
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL DEVELOPMENT & HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  6. MODEL TRAINING")
print("=" * 65)

# ── 6a. REGRESSION MODELS ────────────────────────────────────────────────────

# --- Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_leaf=4,
    random_state=42, n_jobs=-1
)
rf_reg.fit(X_train, y_train_reg)
rf_pred_reg = rf_reg.predict(X_test)
print("  [RF Regressor] trained.")

# --- XGBoost Regressor
xgb_reg = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
xgb_reg.fit(X_train, y_train_reg,
            eval_set=[(X_test, y_test_reg)], verbose=False)
xgb_pred_reg = xgb_reg.predict(X_test)
print("  [XGB Regressor] trained.")

# ── 6b. CLASSIFICATION MODELS ────────────────────────────────────────────────

# --- Logistic Regression (with hyperparameter search)
lr_param_grid = {"C": [0.01, 0.1, 1, 10]}
lr_cv = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_param_grid, cv=5, scoring="f1", n_jobs=-1
)
lr_cv.fit(X_train_sc, y_train_class)
lr_clf = lr_cv.best_estimator_
lr_pred_class = lr_clf.predict(X_test_sc)
print(f"  [LR Classifier] best C = {lr_cv.best_params_['C']}")

# --- Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_train_class)
rf_pred_class = rf_clf.predict(X_test)
print("  [RF Classifier] trained.")

# --- XGBoost Classifier
scale_pos = (y_train_class == 0).sum() / (y_train_class == 1).sum()
xgb_clf = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    scale_pos_weight=scale_pos, use_label_encoder=False,
    eval_metric="logloss", random_state=42, verbosity=0
)
xgb_clf.fit(X_train, y_train_class)
xgb_pred_class = xgb_clf.predict(X_test)
print("  [XGB Classifier] trained.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PERFORMANCE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)}

def evaluate_classification(name, y_true, y_pred):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

reg_results = pd.DataFrame([
    evaluate_regression("Random Forest",  y_test_reg, rf_pred_reg),
    evaluate_regression("XGBoost",        y_test_reg, xgb_pred_reg),
])

clf_results = pd.DataFrame([
    evaluate_classification("Logistic Regression", y_test_class, lr_pred_class),
    evaluate_classification("Random Forest",       y_test_class, rf_pred_class),
    evaluate_classification("XGBoost",             y_test_class, xgb_pred_class),
])

print("\n" + "=" * 65)
print("  REGRESSION RESULTS  (target = Malaria_Cases)")
print("=" * 65)
print(reg_results.to_string(index=False))

print("\n" + "=" * 65)
print("  CLASSIFICATION RESULTS  (target = High_Risk_Binary)")
print("=" * 65)
print(clf_results.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#1a6faf", "#e05c2e", "#2ca02c", "#9467bd"]

# ── 8a. Actual vs Predicted (regression) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, preds, color) in zip(
    axes,
    [("Random Forest", rf_pred_reg, PALETTE[0]),
     ("XGBoost",       xgb_pred_reg, PALETTE[1])]
):
    ax.scatter(y_test_reg, preds, alpha=0.4, color=color, edgecolors="k",
               linewidths=0.3, s=25)
    lims = [y_test_reg.min(), y_test_reg.max()]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Malaria Cases")
    ax.set_ylabel("Predicted Malaria Cases")
    ax.set_title(f"{name} — Actual vs Predicted")
    ax.legend()
fig.suptitle("Regression: Actual vs Predicted Malaria Cases", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/01_actual_vs_predicted.png", dpi=150)
plt.close()
print("  Saved → outputs/01_actual_vs_predicted.png")

# ── 8b. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, preds) in zip(
    axes,
    [("Logistic\nRegression", lr_pred_class),
     ("Random\nForest",       rf_pred_class),
     ("XGBoost",              xgb_pred_class)]
):
    cm = confusion_matrix(y_test_class, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"]).plot(ax=ax, colorbar=False)
    ax.set_title(name, fontsize=11, fontweight="bold")
fig.suptitle("Classification: Confusion Matrices (High-Risk Prediction)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/02_confusion_matrices.png", dpi=150)
plt.close()
print("  Saved → outputs/02_confusion_matrices.png")

# ── 8c. Metric Comparison Bar Charts ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression
ax = axes[0]
x = np.arange(len(reg_results))
w = 0.3
ax.bar(x - w/2, reg_results["MAE"],  w, label="MAE",  color=PALETTE[0])
ax.bar(x + w/2, reg_results["RMSE"], w, label="RMSE", color=PALETTE[1])
ax.set_xticks(x); ax.set_xticklabels(reg_results["Model"])
ax.set_title("Regression Error Comparison"); ax.legend()
ax.set_ylabel("Error (cases)")

# Classification F1 / Accuracy
ax = axes[1]
x = np.arange(len(clf_results))
ax.bar(x - w/2, clf_results["Accuracy"], w, label="Accuracy", color=PALETTE[2])
ax.bar(x + w/2, clf_results["F1"],       w, label="F1 Score", color=PALETTE[3])
ax.set_xticks(x); ax.set_xticklabels(clf_results["Model"])
ax.set_title("Classification Metric Comparison"); ax.legend()
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")

fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_model_comparison.png", dpi=150)
plt.close()
print("  Saved → outputs/03_model_comparison.png")

# ── 8d. Feature Importance (Random Forest Regressor) ─────────────────────────
importances = pd.Series(rf_reg.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
importances.plot.barh(ax=ax, color=PALETTE[0], edgecolor="k", linewidth=0.5)
ax.set_title("Feature Importance — Random Forest Regressor (Top 15)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/04_feature_importance.png", dpi=150)
plt.close()
print("  Saved → outputs/04_feature_importance.png")

# ── 8e. Monthly Case Trend per Region ────────────────────────────────────────
monthly = (
    df.groupby(["Date", "Region"])["Malaria_Cases"]
    .mean().reset_index()
)
fig, ax = plt.subplots(figsize=(14, 5))
for region, color in zip(monthly["Region"].unique(), PALETTE):
    sub = monthly[monthly["Region"] == region].sort_values("Date")
    ax.plot(sub["Date"], sub["Malaria_Cases"], label=region, color=color, lw=1.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=35)
ax.set_title("Average Monthly Malaria Cases by Region (2022–2026)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Avg Malaria Cases"); ax.legend()
plt.tight_layout()
plt.savefig("outputs/05_monthly_trend_by_region.png", dpi=150)
plt.close()
print("  Saved → outputs/05_monthly_trend_by_region.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FUTURE TREND FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  9. FUTURE TREND FORECASTING (12 months ahead per Region)")
print("=" * 65)

FORECAST_MONTHS = 12

# ── 9a. ML-based forecast using best regressor (XGBoost) ────────────────────

def generate_future_features(df_region, future_dates, region_enc, county_enc_mode):
    """Build a synthetic future feature DataFrame for one region."""
    last = df_region.sort_values("Date").tail(3)
    lag1 = last["Malaria_Cases"].iloc[-1]
    lag2 = last["Malaria_Cases"].iloc[-2]
    lag3 = last["Malaria_Cases"].iloc[-3]
    roll = last["Malaria_Cases"].mean()

    rows = []
    for dt in future_dates:
        month = dt.month
        rows.append({
            "Region_enc":        region_enc,
            "County_enc":        county_enc_mode,
            "Year":              dt.year,
            "Month_sin":         np.sin(2 * np.pi * month / 12),
            "Month_cos":         np.cos(2 * np.pi * month / 12),
            "Population":        df_region["Population"].median(),
            "Rainfall_mm":       df_region.groupby("Month")["Rainfall_mm"].mean().get(month, df_region["Rainfall_mm"].mean()),
            "Temperature_C":     df_region.groupby("Month")["Temperature_C"].mean().get(month, df_region["Temperature_C"].mean()),
            "Humidity_percent":  df_region.groupby("Month")["Humidity_percent"].mean().get(month, df_region["Humidity_percent"].mean()),
            "Lag_1_Month_Cases": lag1,
            "Lag_2_Month_Cases": lag2,
            "Lag_3_Month_Cases": lag3,
            "Rolling_3M_Mean":   roll,
            "Rainfall_Temp":     df_region["Rainfall_mm"].mean() * df_region["Temperature_C"].mean(),
            "Humidity_Temp":     df_region["Humidity_percent"].mean() * df_region["Temperature_C"].mean(),
            "Rainfall_Humidity": df_region["Rainfall_mm"].mean() * df_region["Humidity_percent"].mean(),
            "Health_Facilities": df_region["Health_Facilities"].median(),
            "Avg_Income":        df_region["Avg_Income"].median(),
        })
    return pd.DataFrame(rows, index=future_dates)

last_date     = df["Date"].max()
future_dates  = pd.date_range(start=last_date + pd.DateOffset(months=1),
                               periods=FORECAST_MONTHS, freq="MS")

regions      = df["Region"].unique()
region_encs  = df.groupby("Region")["Region_enc"].first().to_dict()
county_modes = df.groupby("Region")["County_enc"].agg(lambda x: x.mode()[0]).to_dict()

forecasts = {}
for region in regions:
    df_r    = df[df["Region"] == region]
    X_fut   = generate_future_features(df_r, future_dates,
                                        region_encs[region], county_modes[region])
    preds   = xgb_reg.predict(X_fut[FEATURE_COLS])
    forecasts[region] = pd.Series(preds, index=future_dates, name=region)

forecast_df = pd.DataFrame(forecasts)
print("\n  12-Month ML Forecast (XGBoost) — avg monthly cases:")
print(forecast_df.round(0).to_string())

# ── 9b. SARIMA time-series forecast (overall Kenya-level) ────────────────────
if SARIMA_AVAILABLE:
    ts = (df.groupby("Date")["Malaria_Cases"].sum()
            .sort_index()
            .asfreq("MS")
            .fillna(method="ffill"))

    sarima = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit    = sarima.fit(disp=False)
    sarima_fc     = sarima_fit.forecast(steps=FORECAST_MONTHS)
    sarima_ci     = sarima_fit.get_forecast(steps=FORECAST_MONTHS).conf_int()

# ── 9c. Forecast visualisation ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left — ML (XGBoost) region-level forecast
ax = axes[0]
historical = df.groupby(["Date", "Region"])["Malaria_Cases"].mean().reset_index()
for region, color in zip(regions, PALETTE):
    hist = historical[historical["Region"] == region].sort_values("Date")
    ax.plot(hist["Date"], hist["Malaria_Cases"],
            color=color, lw=1.4, label=f"{region} (hist.)")
    ax.plot(future_dates, forecast_df[region],
            color=color, lw=2, linestyle="--", marker="o", markersize=4,
            label=f"{region} (forecast)")

ax.axvline(last_date, color="grey", linestyle=":", lw=1.5, label="Forecast start")
ax.set_title("XGBoost Forecast — Malaria Cases per Region\n(next 12 months)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Avg Malaria Cases")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35)
ax.legend(fontsize=7)

# Right — SARIMA national-level forecast
ax = axes[1]
if SARIMA_AVAILABLE:
    ax.plot(ts.index, ts.values, color=PALETTE[0], lw=1.8, label="Historical (Kenya total)")
    ax.plot(sarima_fc.index, sarima_fc.values, color=PALETTE[1],
            lw=2, linestyle="--", marker="o", markersize=4, label="SARIMA Forecast")
    ax.fill_between(sarima_ci.index,
                    sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1],
                    color=PALETTE[1], alpha=0.15, label="95 % CI")
    ax.axvline(last_date, color="grey", linestyle=":", lw=1.5)
    ax.set_title("SARIMA Forecast — Total Kenya Malaria Cases\n(next 12 months)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Cases (summed regions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35)
    ax.legend()
else:
    ax.text(0.5, 0.5, "SARIMA unavailable\n(install statsmodels)",
            ha="center", va="center", fontsize=12, color="grey",
            transform=ax.transAxes)

fig.suptitle("Future Malaria Trend Forecasting — Kenya Endemic Regions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/06_future_forecast.png", dpi=150)
plt.close()
print("  Saved → outputs/06_future_forecast.png")

# ── 9d. Correlation heatmap ───────────────────────────────────────────────────
corr_cols = ["Malaria_Cases", "Rainfall_mm", "Temperature_C",
             "Humidity_percent", "Lag_1_Month_Cases", "Incidence_per_100k"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved → outputs/07_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY REPORT")
print("=" * 65)
best_reg = reg_results.loc[reg_results["R²"].idxmax(), "Model"]
best_clf = clf_results.loc[clf_results["F1"].idxmax(),  "Model"]

print(f"  Best REGRESSION  model  : {best_reg}  (R² = {reg_results.loc[reg_results['Model']==best_reg,'R²'].values[0]})")
print(f"  Best CLASSIFIER  model  : {best_clf}  (F1 = {clf_results.loc[clf_results['Model']==best_clf,'F1'].values[0]})")
print(f"  Forecast horizon        : {FORECAST_MONTHS} months beyond {last_date.strftime('%B %Y')}")
print(f"  Outputs saved in        : ./outputs/")
print("=" * 65)
print("\n  Pipeline complete. ✓")
