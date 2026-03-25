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

    drop_cols = ['ID', 'Health_Facilities', 'Avg_Income', 'Disease_Cases', 'Notes']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.drop_duplicates(inplace=True)

    df['Region'] = df['Region'].str.strip().str.title()
    df['County'] = df['County'].str.strip().str.title()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imp = SimpleImputer(strategy='median')
    df[num_cols] = imp.fit_transform(df[num_cols])

    cap_cols = ['Rainfall_mm', 'Temperature_C', 'Humidity_percent',
                'Malaria_Cases', 'Lag_1_Month_Cases', 'Incidence_per_100k']
    for col in cap_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

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

with tab3:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(models_sel):
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


# ════════════════════════════════════════════════════════════════
# ── Section 1: Predict a New Case (free-form, no fixed limits) ──
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔮 Predict a New Case")
st.markdown(
    "Enter **any values** below — there are no fixed limits. "
    "Type in real figures from your dataset or hypothetical scenarios to see how the models respond."
)

# Helper: derive dataset min/max as soft hints shown in captions
def hint(col):
    if col in X.columns:
        lo, hi = X[col].min(), X[col].max()
        return f"Dataset range: {lo:.1f} – {hi:.1f}"
    return ""

with st.form("predict_form"):
    st.markdown("##### 🌦️ Climate Inputs")
    c1, c2, c3 = st.columns(3)
    rainfall    = c1.number_input("Rainfall (mm)",      value=100.0, step=0.1,
                                   help=hint('Rainfall_mm'))
    temperature = c2.number_input("Temperature (°C)",   value=25.0,  step=0.1,
                                   help=hint('Temperature_C'))
    humidity    = c3.number_input("Humidity (%)",        value=65.0,  step=0.1,
                                   help=hint('Humidity_percent'))

    st.markdown("##### 🦟 Case & Population Inputs")
    c4, c5, c6 = st.columns(3)
    lag_cases   = c4.number_input("Lag 1 Month Cases",  value=500.0, step=1.0,
                                   help=hint('Lag_1_Month_Cases'))
    incidence   = c5.number_input("Incidence per 100k", value=100.0, step=0.1,
                                   help=hint('Incidence_per_100k'))
    month       = c6.number_input("Month (1–12)",        value=6,    step=1,
                                   min_value=1, max_value=12,
                                   help="Calendar month of the observation")

    c7, c8     = st.columns(2)
    population = c7.number_input("Population",          value=500000, step=1000,
                                  help=hint('Population'))
    mal_cases  = c8.number_input("Malaria Cases",       value=500.0,  step=1.0,
                                  help=hint('Malaria_Cases'))

    submitted = st.form_submit_button("🔍 Predict Risk", type="primary")

if submitted:
    cases_pc = mal_cases / max(population, 1) * 100000

    def get_season(m):
        if m in [3,4,5]:    return 'Long_Rains'
        elif m in [6,7,8]:  return 'Dry'
        elif m in [9,10,11]:return 'Short_Rains'
        else:               return 'Cool_Dry'

    season_map = {'Cool_Dry':0, 'Dry':1, 'Long_Rains':2, 'Short_Rains':3}
    season_enc = season_map.get(get_season(month), 0)

    input_df = pd.DataFrame([[
        rainfall, temperature, humidity, lag_cases, incidence,
        month, population, mal_cases, cases_pc, 0, 0, season_enc
    ]], columns=FEATURES)

    st.markdown("### 🎯 Prediction Results")
    pred_cols = st.columns(len(models_sel))
    for i, name in enumerate(models_sel):
        inp   = scaler.transform(input_df) if name == "Logistic Regression" else input_df
        pred  = results[name]['model'].predict(inp)[0]
        prob  = results[name]['model'].predict_proba(inp)[0][1]
        label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
        pred_cols[i].metric(name, label, f"Confidence: {prob*100:.1f}%")

    with st.expander("📋 View input summary"):
        summary = pd.DataFrame({
            'Feature': ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                        'Lag 1 Month Cases', 'Incidence per 100k', 'Month',
                        'Population', 'Malaria Cases', 'Cases per Capita', 'Season'],
            'Value':   [rainfall, temperature, humidity, lag_cases, incidence,
                        month, population, mal_cases, round(cases_pc, 2),
                        get_season(month)]
        })
        st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# ── Section 2: Custom Training Experiment (dataset copy only) ───
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧪 Custom Training Experiment")
st.markdown(
    "Retrain models on a **filtered copy** of your dataset to test how different "
    "data slices affect performance. **The original dataset is never modified.**"
)

with st.expander("⚙️ Configure & Run Custom Experiment", expanded=False):

    st.markdown("##### Step 1 — Filter the dataset copy")
    fc1, fc2 = st.columns(2)

    # Month range filter
    month_range = fc1.slider(
        "Include months", 1, 12, (1, 12),
        help="Keep only rows whose Month falls within this range"
    )

    # Rainfall filter
    rain_min_val = float(X['Rainfall_mm'].min())
    rain_max_val = float(X['Rainfall_mm'].max())
    rain_range = fc2.slider(
        "Rainfall range (mm)",
        rain_min_val, rain_max_val,
        (rain_min_val, rain_max_val),
        help="Keep only rows within this rainfall band"
    )

    fc3, fc4 = st.columns(2)
    temp_min_val = float(X['Temperature_C'].min())
    temp_max_val = float(X['Temperature_C'].max())
    temp_range = fc3.slider(
        "Temperature range (°C)",
        temp_min_val, temp_max_val,
        (temp_min_val, temp_max_val)
    )

    # Sample size
    max_rows = len(X)
    sample_pct = fc4.slider(
        "Use what % of filtered rows",
        10, 100, 100,
        help="Randomly sample this percentage of the filtered dataset for training"
    )

    st.markdown("##### Step 2 — Choose a model")
    exp_model_name = st.selectbox(
        "Model for this experiment",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

    exp_test_size = st.slider("Experiment test split %", 10, 40, 20) / 100

    run_exp = st.button("▶️ Run Custom Experiment", type="primary")

    if run_exp:
        # --- Work on a COPY of X and y, never df_raw ---
        X_exp = X.copy()
        y_exp = y.copy()

        # Apply filters
        mask = (
            (X_exp['Month'] >= month_range[0]) & (X_exp['Month'] <= month_range[1]) &
            (X_exp['Rainfall_mm'] >= rain_range[0]) & (X_exp['Rainfall_mm'] <= rain_range[1]) &
            (X_exp['Temperature_C'] >= temp_range[0]) & (X_exp['Temperature_C'] <= temp_range[1])
        )
        X_exp = X_exp[mask]
        y_exp = y_exp[mask]

        # Sample
        if sample_pct < 100:
            sample_n = max(int(len(X_exp) * sample_pct / 100), 10)
            idx = np.random.RandomState(42).choice(len(X_exp), sample_n, replace=False)
            X_exp = X_exp.iloc[idx]
            y_exp = y_exp.iloc[idx]

        if len(X_exp) < 20:
            st.error("⚠️ Not enough rows after filtering (need at least 20). "
                     "Please widen your filter ranges.")
        elif y_exp.nunique() < 2:
            st.error("⚠️ Filtered data contains only one class — cannot train a classifier. "
                     "Please adjust the filters.")
        else:
            st.info(
                f"🔬 Training on **{len(X_exp)} rows** "
                f"({int((y_exp==0).sum())} low-risk, {int((y_exp==1).sum())} high-risk) "
                f"— original dataset untouched ({len(X)} rows)."
            )

            Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(
                X_exp, y_exp, test_size=exp_test_size, random_state=42, stratify=y_exp
            )

            sc_e = StandardScaler()
            Xtr_e_sc = sc_e.fit_transform(Xtr_e)
            Xte_e_sc = sc_e.transform(Xte_e)

            # Train chosen model (no grid search to keep it fast)
            if exp_model_name == "Logistic Regression":
                m_exp = LogisticRegression(max_iter=1000, random_state=42)
                m_exp.fit(Xtr_e_sc, ytr_e)
                yp_e  = m_exp.predict(Xte_e_sc)
                ypr_e = m_exp.predict_proba(Xte_e_sc)[:, 1]
            elif exp_model_name == "Random Forest":
                m_exp = RandomForestClassifier(n_estimators=100, random_state=42)
                m_exp.fit(Xtr_e, ytr_e)
                yp_e  = m_exp.predict(Xte_e)
                ypr_e = m_exp.predict_proba(Xte_e)[:, 1]
            else:
                m_exp = GradientBoostingClassifier(n_estimators=100, random_state=42)
                m_exp.fit(Xtr_e, ytr_e)
                yp_e  = m_exp.predict(Xte_e)
                ypr_e = m_exp.predict_proba(Xte_e)[:, 1]

            # Metrics
            exp_metrics = {
                'Accuracy':  round(accuracy_score(yte_e, yp_e), 4),
                'Precision': round(precision_score(yte_e, yp_e, zero_division=0), 4),
                'Recall':    round(recall_score(yte_e, yp_e, zero_division=0), 4),
                'F1 Score':  round(f1_score(yte_e, yp_e, zero_division=0), 4),
                'ROC-AUC':   round(roc_auc_score(yte_e, ypr_e), 4),
            }

            st.markdown(f"#### 📊 Results — {exp_model_name} (Custom Experiment)")
            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            for col, (metric, val) in zip([ec1,ec2,ec3,ec4,ec5], exp_metrics.items()):

                # Compare against the same model trained on full data (if available)
                delta_str = None
                if exp_model_name in results:
                    delta_val = val - results[exp_model_name][metric]
                    delta_str = f"{delta_val:+.4f} vs full data"
                col.metric(metric, f"{val:.4f}", delta_str)

            # Side-by-side confusion matrix vs full-data model
            if exp_model_name in results:
                st.markdown("##### Confusion Matrix Comparison")
                cm_cols = st.columns(2)

                for idx_cm, (label_cm, cm_data) in enumerate([
                    (f"{exp_model_name} — Custom ({len(X_exp)} rows)",
                     confusion_matrix(yte_e, yp_e)),
                    (f"{exp_model_name} — Full Data ({len(X)} rows)",
                     results[exp_model_name]['cm']),
                ]):
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    im = ax.imshow(cm_data, interpolation='nearest', cmap='Purples')
                    plt.colorbar(im, ax=ax)
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(['Low Risk','High Risk'], rotation=30, fontsize=9)
                    ax.set_yticklabels(['Low Risk','High Risk'], fontsize=9)
                    thresh = cm_data.max() / 2
                    for r in range(2):
                        for c in range(2):
                            ax.text(c, r, str(cm_data[r, c]),
                                    ha='center', va='center', fontsize=13, fontweight='bold',
                                    color='white' if cm_data[r,c] > thresh else 'black')
                    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                    ax.set_title(label_cm, fontweight='bold', fontsize=9)
                    plt.tight_layout()
                    cm_cols[idx_cm].pyplot(fig)
                    plt.close()
            else:
                st.info("Train the main models first to see a side-by-side comparison.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Group 3 | BSc Data Science | Meru University of Science and Technology | 2026")
