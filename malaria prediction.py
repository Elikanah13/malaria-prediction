# malaria_prediction_app.py
"""
=============================================================================
  MALARIA INFECTION FORECASTING PIPELINE
  Meru University of Science and Technology — Group 3
  Streamlit version (cloud-ready)
=============================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBRegressor, XGBClassifier

# Optional SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False

st.set_page_config(page_title="Malaria Forecasting Pipeline", layout="wide")
st.title("Malaria Infection Forecasting Pipeline")

# ─────────────────────────────────────
# 1. DATA UPLOAD
# ─────────────────────────────────────
uploaded_file = st.file_uploader("Upload Malaria Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded → {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    # ─────────────────────────────────────
    # 2. DATA PREPROCESSING
    # ─────────────────────────────────────
    def preprocess(df):
        df = df.copy()
        # Drop irrelevant columns
        drop_cols = ["ID", "Notes", "Disease_Cases"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Numeric NaNs → median
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Outlier capping (IQR × 3)
        for col in ["Malaria_Cases", "Rainfall_mm", "Incidence_per_100k"]:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 3*iqr, q3 + 3*iqr)

        # Encode categorical
        for c in ["Region", "County"]:
            if c in df.columns:
                le = LabelEncoder()
                df[f"{c}_enc"] = le.fit_transform(df[c])
                df.attrs[f"{c}_classes"] = list(le.classes_)

        # Date column
        df["Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
        )

        return df

    df = preprocess(df)
    st.write("✅ Preprocessing completed.")

    # ─────────────────────────────────────
    # 3. FEATURE ENGINEERING
    # ─────────────────────────────────────
    def engineer_features(df):
        df = df.copy()

        # Cyclical month
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        # Interaction features
        df["Rainfall_Temp"] = df["Rainfall_mm"] * df["Temperature_C"]
        df["Humidity_Temp"] = df["Humidity_percent"] * df["Temperature_C"]
        df["Rainfall_Humidity"] = df["Rainfall_mm"] * df["Humidity_percent"]

        # Lag features
        df.sort_values(["Region", "County", "Year", "Month"], inplace=True)
        grp = df.groupby(["Region", "County"])
        df["Lag_2_Month_Cases"] = grp["Malaria_Cases"].shift(2)
        df["Lag_3_Month_Cases"] = grp["Malaria_Cases"].shift(3)
        df["Rolling_3M_Mean"] = grp["Malaria_Cases"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        lag_cols = ["Lag_2_Month_Cases", "Lag_3_Month_Cases", "Rolling_3M_Mean"]
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())

        # Population-normalized cases
        df["Cases_per_pop"] = df["Malaria_Cases"] / df["Population"]

        return df

    df = engineer_features(df)
    st.write("✅ Feature engineering completed.")

    # ─────────────────────────────────────
    # 4. FEATURE SELECTION
    # ─────────────────────────────────────
    FEATURE_COLS = [
        "Region_enc", "County_enc", "Year", "Month_sin", "Month_cos",
        "Population", "Rainfall_mm", "Temperature_C", "Humidity_percent",
        "Lag_1_Month_Cases", "Lag_2_Month_Cases", "Lag_3_Month_Cases",
        "Rolling_3M_Mean", "Rainfall_Temp", "Humidity_Temp",
        "Rainfall_Humidity", "Health_Facilities", "Avg_Income"
    ]
    TARGET_REG   = "Malaria_Cases"
    TARGET_CLASS = "High_Risk_Binary"

    X = df[FEATURE_COLS]
    y_reg = df[TARGET_REG]
    y_class = df[TARGET_CLASS]

    # ─────────────────────────────────────
    # 5. TRAIN / TEST SPLIT
    # ─────────────────────────────────────
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df  = df_sorted.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]
    y_train_reg = train_df[TARGET_REG]
    y_test_reg  = test_df[TARGET_REG]
    y_train_class = train_df[TARGET_CLASS]
    y_test_class  = test_df[TARGET_CLASS]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    st.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ─────────────────────────────────────
    # 6. MODEL TRAINING
    # ─────────────────────────────────────
    st.write("### Training Models ...")

    # Regression
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)
    xgb_reg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_reg.fit(X_train, y_train_reg, eval_set=[(X_test, y_test_reg)], verbose=False)

    # Classification
    lr_cv = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                         {"C":[0.01,0.1,1,10]}, cv=5, scoring="f1", n_jobs=-1)
    lr_cv.fit(X_train_sc, y_train_class)
    lr_clf = lr_cv.best_estimator_

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_class)

    scale_pos = (y_train_class==0).sum() / (y_train_class==1).sum()
    xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                            scale_pos_weight=scale_pos, use_label_encoder=False,
                            eval_metric="logloss", random_state=42, verbosity=0)
    xgb_clf.fit(X_train, y_train_class)

    st.success(" Model training completed.")

    # ─────────────────────────────────────
    # 7. PERFORMANCE EVALUATION
    # ─────────────────────────────────────
    def evaluate_regression(name, y_true, y_pred):
        return {
            "Model": name,
            "MAE": round(mean_absolute_error(y_true, y_pred),2),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)),2),
            "R²": round(r2_score(y_true, y_pred),4)
        }
    def evaluate_classification(name, y_true, y_pred):
        return {
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred),4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0),4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0),4),
            "F1": round(f1_score(y_true, y_pred, zero_division=0),4)
        }

    reg_results = pd.DataFrame([
        evaluate_regression("Random Forest", y_test_reg, rf_reg.predict(X_test)),
        evaluate_regression("XGBoost", y_test_reg, xgb_reg.predict(X_test))
    ])
    clf_results = pd.DataFrame([
        evaluate_classification("Logistic Regression", y_test_class, lr_clf.predict(X_test_sc)),
        evaluate_classification("Random Forest", y_test_class, rf_clf.predict(X_test)),
        evaluate_classification("XGBoost", y_test_class, xgb_clf.predict(X_test))
    ])

    st.write("### Regression Results")
    st.dataframe(reg_results)
    st.write("### Classification Results")
    st.dataframe(clf_results)

    # ─────────────────────────────────────
    # 8. VISUALISATIONS
    # ─────────────────────────────────────
    st.write("### Regression: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(y_test_reg, rf_reg.predict(X_test), alpha=0.4, label="RF", color="blue")
    ax.scatter(y_test_reg, xgb_reg.predict(X_test), alpha=0.4, label="XGB", color="red")
    lims = [y_test_reg.min(), y_test_reg.max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual Malaria Cases")
    ax.set_ylabel("Predicted Malaria Cases")
    ax.legend()
    st.pyplot(fig)

    st.write("### Classification: Confusion Matrices")
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    for ax, (name, pred) in zip(axes, [("LR", lr_clf.predict(X_test_sc)),
                                       ("RF", rf_clf.predict(X_test)),
                                       ("XGB", xgb_clf.predict(X_test))]):
        cm = confusion_matrix(y_test_class, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    st.pyplot(fig)

    st.success("✅ Visualization complete.")

else:
    st.info("Upload your dataset to start the pipeline.")
