import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
import joblib

st.title("🦟 Malaria Prediction System")
st.write("Train models and predict malaria outbreak risk")

# ===============================
# SIDEBAR SETTINGS
# ===============================

st.sidebar.header("Model Settings")

dataset_file = st.sidebar.file_uploader(
    "Upload Malaria Dataset", 
    type=["csv"]
)

train_button = st.sidebar.button("Train Models")

# ===============================
# MODEL TRAINING
# ===============================

if dataset_file is not None:

    df = pd.read_csv(dataset_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    if train_button:

        st.write("Training models...")

        target = "High_Risk_Binary"

        X = df.drop(columns=[target])
        y = df[target]

        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ])

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "Random Forest": RandomForestClassifier(n_estimators=200),

            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                eval_metric="logloss"
            )
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_acc = 0
        best_model = None
        best_name = ""

        for name, model in models.items():

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)

            st.write(f"{name} Accuracy: {acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                best_model = pipe
                best_name = name

        st.success(f"Best Model: {best_name} (Accuracy {best_acc:.3f})")

        joblib.dump(best_model, "malaria_model.pkl")

        st.success("Model saved successfully")

# ===============================
# PREDICTION SECTION
# ===============================

st.header("Predict Malaria Risk")

try:
    model = joblib.load("malaria_model.pkl")

    region = st.selectbox("Region", ["Nyanza","Rift Valley","Central"])

    rainfall = st.slider("Rainfall (mm)", 0, 500, 120)

    temperature = st.slider("Temperature (°C)", 10, 40, 26)

    humidity = st.slider("Humidity (%)", 0, 100, 75)

    month = st.selectbox(
        "Month",
        ["Jan","Feb","Mar","Apr","May","Jun",
         "Jul","Aug","Sep","Oct","Nov","Dec"]
    )

    input_df = pd.DataFrame({
        "Region":[region],
        "Rainfall":[rainfall],
        "Temperature":[temperature],
        "Humidity":[humidity],
        "Month":[month]
    })

    st.write("Input Data")
    st.write(input_df)

    if st.button("Predict"):

        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("⚠ High Malaria Risk Predicted")
        else:
            st.success("✅ Low Malaria Risk")

except:
    st.warning("Please train the model first using the sidebar.")
