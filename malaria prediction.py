import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBRegressor, XGBClassifier

# Optional SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    st.warning("statsmodels not installed — SARIMA forecasting skipped.")

# ───────────────────────────────
# STREAMLIT APP TITLE
# ───────────────────────────────
st.title("Malaria Infection Forecasting Pipeline")
st.write("Meru University of Science and Technology — Group 3, Data Science Project")

# ───────────────────────────────
# DATA UPLOADER
# ───────────────────────────────
uploaded_file = st.file_uploader("Upload your Malaria dataset CSV", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# ───────────────────────────────
# DATA LOADING
# ───────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    st.write(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns and types:")
    st.write(df.dtypes)
    missing = df.isnull().sum()
    if missing.any():
        st.write("Missing values per column:")
        st.write(missing[missing > 0])
    return df

df = load_data(uploaded_file)

# ───────────────────────────────
# DATA PREPROCESSING
# ───────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop irrelevant columns if present
    drop_cols = ["ID", "Notes", "Disease_Cases"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Impute numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Encode categorical
    for c in ["Region", "County"]:
        if c in df.columns:
            le = LabelEncoder()
            df[f"{c}_enc"] = le.fit_transform(df[c])
            df.attrs[f"{c}_classes"] = list(le.classes_)

    # Robust date creation
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Year" in df.columns and "Month" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
        df = df.dropna(subset=["Year","Month"])
        df["Month"] = df["Month"].clip(1,12)
        df["Date"] = pd.to_datetime(
            df["Year"].astype(int).astype(str) + "-" +
            df["Month"].astype(int).astype(str).str.zfill(2) + "-01"
        )
    else:
        st.warning("No Date or Year/Month column found — skipping date creation.")

    return df

df = preprocess(df)
st.write("Preprocessing complete. Sample data:")
st.dataframe(df.head())

# ───────────────────────────────
# FEATURE ENGINEERING
# ───────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    if "Lag_1_Month_Cases" in df.columns:
        grp = df.groupby(["Region", "County"]) if "Region" in df.columns and "County" in df.columns else df
        df["Lag_2_Month_Cases"] = grp["Malaria_Cases"].shift(2)
        df["Lag_3_Month_Cases"] = grp["Malaria_Cases"].shift(3)
        df["Rolling_3M_Mean"]   = grp["Malaria_Cases"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df[["Lag_2_Month_Cases","Lag_3_Month_Cases","Rolling_3M_Mean"]] = \
            df[["Lag_2_Month_Cases","Lag_3_Month_Cases","Rolling_3M_Mean"]].fillna(df[["Lag_2_Month_Cases","Lag_3_Month_Cases","Rolling_3M_Mean"]].median())

    st.write("Feature engineering complete. Total features:", df.shape[1])
    return df

df = engineer_features(df)

st.success("✅ Dataset ready for modeling!")

# You can now add model training, evaluation, and visualization as needed.
# Using Streamlit, you can show results via st.write, st.line_chart, st.bar_chart, etc.
