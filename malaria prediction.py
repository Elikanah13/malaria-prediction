"""
Malaria Infection Prediction App
Models: Logistic Regression | Random Forest | Gradient Boosting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Prediction | Group 3",
    page_icon="",
    layout="wide"
)

st.title(" Malaria Infection Prediction")
st.markdown("**Group 3 · Meru University of Science and Technology · BSc Data Science**")
st.markdown("---")

# ── Sidebar — upload ─────────────────────────────────────────
st.sidebar.header(" Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded:
    df_raw = load_data(uploaded)
else:
    st.info(" Please upload **Final_Malaria_Dataset.csv** in the sidebar to begin.")
    st.stop()

st.subheader("📋 Raw Data Preview")
st.dataframe(df_raw.head(10), use_container_width=True)
st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ── Pre-processing ────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()

    # Drop sparse / non-feature columns
    drop_cols = ['ID', 'Health_Facilities', 'Avg_Income', 'Disease_Cases', 'Notes']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Standardise strings
    df['Region'] = df['Region'].str.strip().str.title()
    df['County'] = df['County'].str.strip().str.title()

    # Impute numerics with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imp = SimpleImputer(strategy='median')
    df[num_cols] = imp.fit_transform(df[num_cols])

    # Outlier capping (IQR)
    cap_cols = ['Rainfall_mm', 'Temperature_C', 'Humidity_percent',
                'Malaria_Cases', 'Lag_1_Month_Cases', 'Incidence_per_100k']
    for col in cap_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # Feature engineering
    def get_season(m):
        if m in [3,4,5]:    return 'Long_Rains'
        elif m in [6,7,8]:  return 'Dry'
        elif m in [9,10,11]:return 'Short_Rains'
        else:               return 'Cool_Dry'

    df['Season'] = df['Month'].apply(get_season)
    df['Cases_Per_Capita'] = df['Malaria_Cases'] / df['Population'] * 100000

    le_r = LabelEncoder(); le_c = LabelEncoder(); le_s = LabelEncoder()
    df['Region_enc'] = le_r.fit_transform(df['Region'])
    df['County_enc'] = le_c.fit_transform(df['County'])
    df['Season_enc'] = le_s.fit_transform(df['Season'])

    FEATURES = ['Rainfall_mm', 'Temperature_C', 'Humidity_percent',
                'Lag_1_Month_Cases', 'Incidence_per_100k', 'Month',
                'Population', 'Malaria_Cases', 'Cases_Per_Capita',
                'Region_enc', 'County_enc', 'Season_enc']
    TARGET = 'High_Risk_Binary'

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return X, y, FEATURES

X, y, FEATURES = preprocess(df_raw)

# ── Data stats ───────────────────────────────────────────────
st.subheader("📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records",  len(X))
col2.metric("Low Risk (0)",   int((y == 0).sum()))
col3.metric("High Risk (1)",  int((y == 1).sum()))

with st.expander("Show feature summary"):
    st.dataframe(X.describe().round(2), use_container_width=True)

# ── Sidebar — model options ───────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
test_size   = st.sidebar.slider("Test split %", 10, 40, 20) / 100
run_tuning  = st.sidebar.checkbox("Enable hyperparameter tuning (GridSearchCV)", value=True)
models_sel  = st.sidebar.multiselect(
    "Models to train",
    ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
)

if not models_sel:
    st.warning("Please select at least one model.")
    st.stop()

run_btn = st.sidebar.button(" Train Models", type="primary")

if not run_btn:
    st.info("Configure settings in the sidebar and click **Train Models** to start.")
    st.stop()

# ── Training ─────────────────────────────────────────────────
st.subheader("🔧 Model Training")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_model(name, tuning):
    if name == "Logistic Regression":
        base = LogisticRegression(max_iter=1000, random_state=42)
        if tuning:
            gs = GridSearchCV(base, {'C':[0.01,0.1,1,10], 'solver':['lbfgs','liblinear']},
                              cv=cv, scoring='f1', n_jobs=-1)
            gs.fit(X_train_sc, y_train)
            return gs.best_estimator_, gs.best_params_, X_test_sc
        else:
            base.fit(X_train_sc, y_train)
            return base, {}, X_test_sc

    elif name == "Random Forest":
        base = RandomForestClassifier(random_state=42)
        if tuning:
            gs = GridSearchCV(base,
                {'n_estimators':[100,200], 'max_depth':[5,10,None], 'min_samples_split':[2,5]},
                cv=cv, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)
            return gs.best_estimator_, gs.best_params_, X_test
        else:
            base.fit(X_train, y_train)
            return base, {}, X_test

    elif name == "Gradient Boosting":
        base = GradientBoostingClassifier(random_state=42)
        if tuning:
            gs = GridSearchCV(base,
                {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]},
                cv=cv, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)
            return gs.best_estimator_, gs.best_params_, X_test
        else:
            base.fit(X_train, y_train)
            return base, {}, X_test

results = {}
progress = st.progress(0)
status   = st.empty()

for i, name in enumerate(models_sel):
    status.text(f"Training {name}...")
    model, params, Xte = get_model(name, run_tuning)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    results[name] = {
        'model':     model,
        'params':    params,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1 Score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'cm':        confusion_matrix(y_test, y_pred),
    }
    progress.progress((i + 1) / len(models_sel))

status.success("✅ All models trained!")

# ── Results table ─────────────────────────────────────────────
st.subheader("📈 Performance Summary")

metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
summary_df  = pd.DataFrame({
    name: {m: results[name][m] for m in metric_cols}
    for name in models_sel
}).T

best_model = summary_df['F1 Score'].idxmax()
st.dataframe(
    summary_df.style.highlight_max(axis=0, color='#d4edda'),
    use_container_width=True
)
st.success(f" **Best Model: {best_model}** — F1 Score = {summary_df.loc[best_model,'F1 Score']:.4f}")

if run_tuning:
    with st.expander("Best Hyperparameters"):
        for name in models_sel:
            st.write(f"**{name}:** {results[name]['params']}")

# ── Charts ────────────────────────────────────────────────────
st.subheader("📊 Visualisations")

tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Confusion Matrices", "ROC Curves"])

COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

# Tab 1 — Bar chart
with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(metric_cols))
    width = 0.8 / len(models_sel)
    for i, name in enumerate(models_sel):
        vals = [results[name][m] for m in metric_cols]
        bars = ax.bar(x + i*width - (len(models_sel)-1)*width/2,
                      vals, width, label=name, color=COLORS[i], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

# Tab 2 — Confusion matrices
with tab2:
    cols = st.columns(len(models_sel))
    for i, name in enumerate(models_sel):
        cm = results[name]['cm']
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Low Risk', 'High Risk'], rotation=30, fontsize=9)
        ax.set_yticklabels(['Low Risk', 'High Risk'], fontsize=9)
        thresh = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        color='white' if cm[r, c] > thresh else 'black',
                        fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_title(name, fontweight='bold', fontsize=11)
        plt.tight_layout()
        cols[i].pyplot(fig)
        plt.close()

# Tab 3 — ROC curves
with tab3:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(models_sel):
        Xte = X_test_sc if name == "Logistic Regression" else X_test
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
        auc = results[name]['ROC-AUC']
        ax.plot(fpr, tpr, lw=2.2, color=COLORS[i], label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ── Feature importance ────────────────────────────────────────
if "Random Forest" in results:
    st.subheader("🌲 Feature Importance (Random Forest)")
    fi = pd.Series(results['Random Forest']['model'].feature_importances_,
                   index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi.index, fi.values, color='#10B981', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Importance Score')
    ax.set_title('Random Forest — Feature Importance', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ── Live prediction form ──────────────────────────────────────
st.subheader("🔮 Predict a New Case")
st.markdown("Enter climate and case data below to predict malaria risk:")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    rainfall    = c1.number_input("Rainfall (mm)",       0.0, 500.0, 100.0)
    temperature = c2.number_input("Temperature (°C)",    10.0, 40.0,  25.0)
    humidity    = c3.number_input("Humidity (%)",         0.0, 100.0, 65.0)

    c4, c5, c6 = st.columns(3)
    lag_cases   = c4.number_input("Lag 1 Month Cases",   0.0, 5000.0, 500.0)
    incidence   = c5.number_input("Incidence per 100k",  0.0, 1000.0, 100.0)
    month       = c6.slider("Month", 1, 12, 6)

    c7, c8     = st.columns(2)
    population = c7.number_input("Population",     10000, 5000000, 500000)
    mal_cases  = c8.number_input("Malaria Cases",  0.0,   10000.0, 500.0)

    submitted = st.form_submit_button("Predict Risk", type="primary")

if submitted:
    cases_pc = mal_cases / population * 100000

    def get_season(m):
        if m in [3,4,5]:    return 'Long_Rains'
        elif m in [6,7,8]:  return 'Dry'
        elif m in [9,10,11]:return 'Short_Rains'
        else:               return 'Cool_Dry'

    season_map = {'Cool_Dry':0, 'Dry':1, 'Long_Rains':2, 'Short_Rains':3}
    season_enc = season_map.get(get_season(month), 0)

    input_df = pd.DataFrame([[
        rainfall, temperature, humidity, lag_cases, incidence,
        month, population, mal_cases, cases_pc,
        0, 0, season_enc
    ]], columns=FEATURES)

    st.markdown("### 🎯 Prediction Results")
    pred_cols = st.columns(len(models_sel))
    for i, name in enumerate(models_sel):
        if name == "Logistic Regression":
            inp = scaler.transform(input_df)
        else:
            inp = input_df
        pred  = results[name]['model'].predict(inp)[0]
        prob  = results[name]['model'].predict_proba(inp)[0][1]
        label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
        pred_cols[i].metric(name, label, f"Confidence: {prob*100:.1f}%")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Group 3 | BSc Data Science | Meru University of Science and Technology | 2026")
