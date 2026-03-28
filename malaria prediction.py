"""
Malaria Infection Prediction App
Models: Logistic Regression | Random Forest | Gradient Boosting
+ Future Trend Forecasting (Cases & High-Risk Probability)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error
)
from scipy import stats

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Prediction | Group 3",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Malaria Infection Prediction")
st.markdown("*Group 3 · Meru University of Science and Technology · BSc Data Science*")
st.markdown("---")

# ── Initialise session state ─────────────────────────────────
# This is the KEY fix: persist trained models across Streamlit reruns
for key in ['results', 'scaler', 'y_test', 'models_sel',
            'X', 'y', 'FEATURES', 'df_processed']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar — upload ─────────────────────────────────────────
st.sidebar.header("📂 Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded:
    df_raw = load_data(uploaded)
else:
    st.info("📁 Please upload *Final_Malaria_Dataset.csv* in the sidebar to begin.")
    st.stop()

st.subheader("📋 Raw Data Preview")
st.dataframe(df_raw.head(10), use_container_width=True)
st.markdown(f"*Shape:* {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

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
        if m in [3,4,5]:     return 'Long_Rains'
        elif m in [6,7,8]:   return 'Dry'
        elif m in [9,10,11]: return 'Short_Rains'
        else:                return 'Cool_Dry'

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
    return X, y, FEATURES, df

X, y, FEATURES, df_processed = preprocess(df_raw)

# Store in session state for prediction access
st.session_state['X'] = X
st.session_state['y'] = y
st.session_state['FEATURES'] = FEATURES
st.session_state['df_processed'] = df_processed

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
test_size  = st.sidebar.slider("Test split %", 10, 40, 20) / 100
run_tuning = st.sidebar.checkbox("Enable hyperparameter tuning (GridSearchCV)", value=True)
models_sel = st.sidebar.multiselect(
    "Models to train",
    ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
)

if not models_sel:
    st.warning("Please select at least one model.")
    st.stop()

run_btn = st.sidebar.button("🚀 Train Models", type="primary")

# ── Training ─────────────────────────────────────────────────
# KEY FIX: only train when button is clicked; results persist in session_state
if run_btn:
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
                gs = GridSearchCV(base,
                    {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
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
                    {'n_estimators': [100, 200], 'max_depth': [5, 10, None],
                     'min_samples_split': [2, 5]},
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
                    {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                     'max_depth': [3, 5]},
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

    # ── Persist everything in session state ──────────────────
    st.session_state['results']    = results
    st.session_state['scaler']     = scaler
    st.session_state['y_test']     = y_test
    st.session_state['models_sel'] = models_sel

# ── Only show results / prediction if models have been trained ─
if st.session_state['results'] is None:
    st.info("Configure settings in the sidebar and click **🚀 Train Models** to begin.")
    st.stop()

# Recover from session state
results    = st.session_state['results']
scaler     = st.session_state['scaler']
y_test     = st.session_state['y_test']
models_sel = st.session_state['models_sel']

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
st.success(f"🏆 **Best Model: {best_model}** — F1 Score = {summary_df.loc[best_model,'F1 Score']:.4f}")

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
# ── Section 1: Predict a New Case ───────────────────────────────
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔮 Predict a New Case")
st.markdown(
    "Enter any values below. The dataset range is shown as a hint in each field."
)

def hint(col):
    if col in X.columns:
        lo, hi = X[col].min(), X[col].max()
        return f"Dataset range: {lo:.1f} – {hi:.1f}"
    return ""

def get_season(m):
    if m in [3,4,5]:     return 'Long_Rains'
    elif m in [6,7,8]:   return 'Dry'
    elif m in [9,10,11]: return 'Short_Rains'
    else:                return 'Cool_Dry'

season_map = {'Cool_Dry': 0, 'Dry': 1, 'Long_Rains': 2, 'Short_Rains': 3}

with st.form("predict_form"):
    st.markdown("##### 🌦️ Climate Inputs")
    c1, c2, c3 = st.columns(3)
    rainfall    = c1.number_input("Rainfall (mm)",      value=100.0, step=0.1, help=hint('Rainfall_mm'))
    temperature = c2.number_input("Temperature (°C)",   value=25.0,  step=0.1, help=hint('Temperature_C'))
    humidity    = c3.number_input("Humidity (%)",        value=65.0,  step=0.1, help=hint('Humidity_percent'))

    st.markdown("##### 🦟 Case & Population Inputs")
    c4, c5, c6 = st.columns(3)
    lag_cases   = c4.number_input("Lag 1 Month Cases",  value=500.0, step=1.0,  help=hint('Lag_1_Month_Cases'))
    incidence   = c5.number_input("Incidence per 100k", value=100.0, step=0.1,  help=hint('Incidence_per_100k'))
    month       = c6.number_input("Month (1–12)",        value=6, step=1, min_value=1, max_value=12)

    c7, c8 = st.columns(2)
    population = c7.number_input("Population",    value=500000, step=1000, help=hint('Population'))
    mal_cases  = c8.number_input("Malaria Cases", value=500.0,  step=1.0,  help=hint('Malaria_Cases'))

    submitted = st.form_submit_button("🔍 Predict Risk", type="primary")

# KEY FIX: prediction block is OUTSIDE and AFTER the form block
# 'submitted' is True on the rerun triggered by clicking the button
if submitted:
    cases_pc   = mal_cases / max(population, 1) * 100000
    season_enc = season_map.get(get_season(int(month)), 0)

    input_df = pd.DataFrame([[
        rainfall, temperature, humidity, lag_cases, incidence,
        month, population, mal_cases, cases_pc, 0, 0, season_enc
    ]], columns=FEATURES)

    st.markdown("### 🎯 Prediction Results")
    pred_cols = st.columns(len(models_sel))

    for i, name in enumerate(models_sel):
        # Logistic Regression needs scaled input; tree models use raw DataFrame
        if name == "Logistic Regression":
            inp = scaler.transform(input_df)
        else:
            inp = input_df

        model = results[name]['model']
        pred  = model.predict(inp)[0]
        prob  = model.predict_proba(inp)[0][1]
        label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
        pred_cols[i].metric(name, label, f"Confidence: {prob*100:.1f}%")

    with st.expander("📋 View input summary"):
        summary = pd.DataFrame({
            'Feature': ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                        'Lag 1 Month Cases', 'Incidence per 100k', 'Month',
                        'Population', 'Malaria Cases', 'Cases per Capita', 'Season'],
            'Value':   [rainfall, temperature, humidity, lag_cases, incidence,
                        month, population, mal_cases, round(cases_pc, 2),
                        get_season(int(month))]
        })
        st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ── Section 2: Future Trend Prediction ──────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📅 Future Trend Prediction")
st.markdown(
    "Forecast *malaria cases* and *high-risk probability* for upcoming months "
    "using time-series regression trained on your historical data."
)

@st.cache_data
def build_time_series(df):
    d = df.copy()
    has_year = 'Year' in d.columns
    if has_year:
        d['time_idx'] = (d['Year'] - d['Year'].min()) * 12 + (d['Month'] - 1)
        ts = (d.groupby('time_idx')
               .agg(
                   Malaria_Cases  =('Malaria_Cases',      'mean'),
                   Incidence      =('Incidence_per_100k', 'mean'),
                   Rainfall       =('Rainfall_mm',        'mean'),
                   Temperature    =('Temperature_C',      'mean'),
                   Humidity       =('Humidity_percent',   'mean'),
                   High_Risk_Rate =('High_Risk_Binary',   'mean'),
                   Month          =('Month',              'first'),
               )
               .reset_index())
    else:
        ts = (d.groupby('Month')
               .agg(
                   Malaria_Cases  =('Malaria_Cases',      'mean'),
                   Incidence      =('Incidence_per_100k', 'mean'),
                   Rainfall       =('Rainfall_mm',        'mean'),
                   Temperature    =('Temperature_C',      'mean'),
                   Humidity       =('Humidity_percent',   'mean'),
                   High_Risk_Rate =('High_Risk_Binary',   'mean'),
               )
               .reset_index()
               .rename(columns={'Month': 'time_idx'}))
        ts['Month'] = ts['time_idx']

    ts = ts.sort_values('time_idx').reset_index(drop=True)
    ts['t'] = np.arange(len(ts))
    return ts, has_year


@st.cache_data
def build_forecast_features(ts):
    t     = ts['t'].values
    month = ts['Month'].values

    sin1 = np.sin(2 * np.pi * month / 12)
    cos1 = np.cos(2 * np.pi * month / 12)
    sin2 = np.sin(4 * np.pi * month / 12)
    cos2 = np.cos(4 * np.pi * month / 12)

    feats = np.column_stack([
        t, t**2, sin1, cos1, sin2, cos2,
        ts['Rainfall'].values, ts['Temperature'].values, ts['Humidity'].values,
    ])
    return feats


def make_future_feats(ts, n_ahead):
    t_last     = ts['t'].max()
    t_fut      = np.arange(t_last + 1, t_last + 1 + n_ahead)
    last_month = int(ts['Month'].iloc[-1])
    months_fut = np.array([(last_month + i - 1) % 12 + 1 for i in range(1, n_ahead + 1)])

    sin1 = np.sin(2 * np.pi * months_fut / 12)
    cos1 = np.cos(2 * np.pi * months_fut / 12)
    sin2 = np.sin(4 * np.pi * months_fut / 12)
    cos2 = np.cos(4 * np.pi * months_fut / 12)

    window = min(12, len(ts))
    rain_f = np.full(n_ahead, ts['Rainfall'].iloc[-window:].mean())
    temp_f = np.full(n_ahead, ts['Temperature'].iloc[-window:].mean())
    hum_f  = np.full(n_ahead, ts['Humidity'].iloc[-window:].mean())

    feats = np.column_stack([t_fut, t_fut**2, sin1, cos1, sin2, cos2, rain_f, temp_f, hum_f])
    return feats, months_fut


ts, has_year = build_time_series(df_processed)

with st.expander("📊 Historical Monthly Aggregated Data", expanded=False):
    st.dataframe(ts.round(2), use_container_width=True)
    st.caption(
        f"{'Year × Month' if has_year else 'Month-only'} aggregation · {len(ts)} time steps"
    )

st.markdown("#### ⚙️ Forecast Settings")
fc1, fc2, fc3 = st.columns(3)
n_ahead     = fc1.slider("Months to forecast ahead", 3, 24, 12)
conf_int    = fc2.checkbox("Show 95% confidence band", value=True)
show_decomp = fc3.checkbox("Show trend decomposition", value=False)

run_forecast = st.button("📈 Generate Forecast", type="primary")

if run_forecast:
    if len(ts) < 6:
        st.error("⚠️ Need at least 6 time steps in the dataset to build a reliable forecast.")
    else:
        X_ts   = build_forecast_features(ts)
        y_cas  = ts['Malaria_Cases'].values
        y_risk = ts['High_Risk_Rate'].values

        gb_cas = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42)
        gb_cas.fit(X_ts, y_cas)

        ridge_risk = Ridge(alpha=1.0)
        ridge_risk.fit(X_ts, y_risk)

        cas_pred_is  = gb_cas.predict(X_ts)
        risk_pred_is = ridge_risk.predict(X_ts).clip(0, 1)

        mae_cas  = mean_absolute_error(y_cas, cas_pred_is)
        rmse_cas = np.sqrt(mean_squared_error(y_cas, cas_pred_is))
        mae_risk = mean_absolute_error(y_risk, risk_pred_is)

        X_fut, months_fut = make_future_feats(ts, n_ahead)
        cas_fut  = gb_cas.predict(X_fut)
        risk_fut = ridge_risk.predict(X_fut).clip(0, 1)

        if conf_int:
            n_boot = 200
            rng = np.random.RandomState(42)
            cas_boots  = np.zeros((n_boot, n_ahead))
            risk_boots = np.zeros((n_boot, n_ahead))
            residuals_cas  = y_cas - cas_pred_is
            residuals_risk = y_risk - risk_pred_is
            for b in range(n_boot):
                cas_boots[b]  = cas_fut + rng.choice(residuals_cas, size=n_ahead, replace=True)
                risk_boots[b] = np.clip(risk_fut + rng.choice(residuals_risk, size=n_ahead, replace=True), 0, 1)
            cas_lo,  cas_hi  = np.percentile(cas_boots,  [2.5, 97.5], axis=0)
            risk_lo, risk_hi = np.percentile(risk_boots, [2.5, 97.5], axis=0)

        future_steps = np.arange(ts['t'].max() + 1, ts['t'].max() + 1 + n_ahead)
        forecast_df  = pd.DataFrame({
            'Step':              future_steps,
            'Month':             months_fut,
            'Forecasted_Cases':  np.round(cas_fut, 1),
            'Forecasted_Risk_%': np.round(risk_fut * 100, 2),
            'Risk_Label':        ['🔴 High' if r >= 0.5 else '🟢 Low' for r in risk_fut],
        })

        st.markdown("#### 📋 Forecast Quality (In-Sample Fit)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cases MAE",     f"{mae_cas:.1f}")
        mc2.metric("Cases RMSE",    f"{rmse_cas:.1f}")
        mc3.metric("Risk Rate MAE", f"{mae_risk:.4f}")

        # ── Plot 1 ───────────────────────────────────────────────
        st.markdown("#### 📉 Malaria Cases — Historical + Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        hist_t = ts['t'].values
        ax.plot(hist_t, y_cas, 'o-', color='#3B82F6', lw=2, markersize=5, label='Historical (actual)', zorder=3)
        ax.plot(hist_t, cas_pred_is, '--', color='#94A3B8', lw=1.5, label='In-sample fit', alpha=0.7)
        ax.plot(future_steps, cas_fut, 's-', color='#EF4444', lw=2.5, markersize=6,
                label=f'Forecast (+{n_ahead} months)', zorder=3)
        if conf_int:
            ax.fill_between(future_steps, cas_lo, cas_hi, color='#EF4444', alpha=0.15, label='95% CI')
        ax.axvline(x=hist_t[-1] + 0.5, color='grey', linestyle=':', lw=1.5, alpha=0.7)
        ax.set_xlabel('Time Step (months elapsed)', fontsize=11)
        ax.set_ylabel('Avg Malaria Cases', fontsize=11)
        ax.set_title('Malaria Cases Trend Forecast', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Plot 2 ───────────────────────────────────────────────
        st.markdown("#### 🔴 High-Risk Probability — Historical + Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_t, y_risk * 100, 'o-', color='#10B981', lw=2, markersize=5,
                label='Historical high-risk rate', zorder=3)
        ax.plot(hist_t, risk_pred_is * 100, '--', color='#94A3B8', lw=1.5, label='In-sample fit', alpha=0.7)
        ax.plot(future_steps, risk_fut * 100, 's-', color='#F59E0B', lw=2.5, markersize=6,
                label=f'Forecast (+{n_ahead} months)', zorder=3)
        if conf_int:
            ax.fill_between(future_steps, risk_lo * 100, risk_hi * 100,
                            color='#F59E0B', alpha=0.15, label='95% CI')
        ax.axhline(50, color='#EF4444', lw=1, linestyle='--', alpha=0.5)
        ax.text(0.01, 51, 'High-risk threshold (50%)', transform=ax.get_xaxis_transform(),
                fontsize=8, color='#EF4444')
        ax.axvline(x=hist_t[-1] + 0.5, color='grey', linestyle=':', lw=1.5, alpha=0.7)
        ax.set_xlabel('Time Step (months elapsed)', fontsize=11)
        ax.set_ylabel('High-Risk Rate (%)', fontsize=11)
        ax.set_ylim(-5, 110)
        ax.set_title('High-Risk Probability Trend Forecast', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Plot 3 ───────────────────────────────────────────────
        st.markdown("#### 🌡️ Seasonal Pattern — Average Cases by Calendar Month")
        month_avg   = ts.groupby('Month')['Malaria_Cases'].mean().reindex(range(1, 13))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(1, 13), month_avg.values,
                      color=['#3B82F6' if v < month_avg.mean() else '#EF4444'
                             for v in month_avg.values],
                      edgecolor='white', alpha=0.9)
        ax.axhline(month_avg.mean(), color='grey', lw=1.5, linestyle='--', label='Annual average')
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_names, fontsize=10)
        ax.set_ylabel('Avg Malaria Cases', fontsize=11)
        ax.set_title('Seasonal Pattern (Calendar Month)', fontweight='bold', fontsize=12)
        ax.legend(); ax.spines[['top','right']].set_visible(False)
        for bar, val in zip(bars, month_avg.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Trend decomposition ──────────────────────────────────
        if show_decomp and len(ts) >= 12:
            st.markdown("#### 🔬 Trend Decomposition")
            t_arr = ts['t'].values.astype(float)
            y_arr = ts['Malaria_Cases'].values.astype(float)
            slope, intercept, r_val, p_val, _ = stats.linregress(t_arr, y_arr)
            trend_line = slope * t_arr + intercept
            seasonal   = y_arr - trend_line
            residual   = y_arr - trend_line - seasonal

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            axes[0].plot(t_arr, y_arr, color='#3B82F6', lw=2)
            axes[0].plot(t_arr, trend_line, color='#EF4444', lw=1.5, linestyle='--', label='Trend')
            axes[0].set_ylabel('Original'); axes[0].legend()
            axes[0].set_title('Trend Decomposition — Malaria Cases', fontweight='bold')
            axes[1].bar(t_arr, seasonal, color='#10B981', alpha=0.7)
            axes[1].axhline(0, color='grey', lw=0.8); axes[1].set_ylabel('Seasonal')
            axes[2].plot(t_arr, residual, color='#F59E0B', lw=1.2)
            axes[2].axhline(0, color='grey', lw=0.8)
            axes[2].set_ylabel('Residual'); axes[2].set_xlabel('Time Step')
            for ax in axes: ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            trend_dir = "📈 Upward" if slope > 0 else "📉 Downward"
            st.info(
                f"**Linear trend slope:** {slope:+.2f} cases/month  ·  "
                f"**Direction:** {trend_dir}  ·  **R²:** {r_val**2:.3f}  ·  **p-value:** {p_val:.4f}"
            )

        # ── Forecast table ───────────────────────────────────────
        st.markdown("#### 📋 Forecast Table")
        if conf_int:
            forecast_df['Cases_Lower_95']   = np.round(cas_lo, 1)
            forecast_df['Cases_Upper_95']   = np.round(cas_hi, 1)
            forecast_df['Risk_%_Lower_95']  = np.round(risk_lo * 100, 2)
            forecast_df['Risk_%_Upper_95']  = np.round(risk_hi * 100, 2)

        st.dataframe(
            forecast_df.style.map(
                lambda v: 'background-color: #fee2e2' if v == '🔴 High'
                          else ('background-color: #dcfce7' if v == '🟢 Low' else ''),
                subset=['Risk_Label']
            ),
            use_container_width=True
        )

        high_risk_months = int((risk_fut >= 0.5).sum())
        st.info(
            f"📊 **Forecast summary:** Over the next **{n_ahead} months**, "
            f"**{high_risk_months}** months are predicted **high-risk** "
            f"and **{n_ahead - high_risk_months}** months **low-risk**. "
            f"Average forecasted cases: **{cas_fut.mean():.1f}** / month."
        )

        peak_idx        = np.argmax(cas_fut)
        month_names_all = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        peak_month_name = month_names_all[months_fut[peak_idx] - 1]
        st.warning(
            f"⚠️ **Highest forecasted burden:** Month **{peak_idx + 1}** of the forecast "
            f"(calendar month: **{peak_month_name}**) — "
            f"predicted **{cas_fut[peak_idx]:.0f} cases** "
            f"with **{risk_fut[peak_idx]*100:.1f}% high-risk probability**."
        )


# ════════════════════════════════════════════════════════════════
# ── Section 3: Custom Training Experiment ───────────────────────
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧪 Custom Training Experiment")
st.markdown(
    "Retrain models on a *filtered copy* of your dataset to test how different "
    "data slices affect performance. *The original dataset is never modified.*"
)

with st.expander("⚙️ Configure & Run Custom Experiment", expanded=False):

    st.markdown("##### Step 1 — Filter the dataset copy")
    fc1, fc2 = st.columns(2)

    month_range = fc1.slider("Include months", 1, 12, (1, 12))

    rain_min_val = float(X['Rainfall_mm'].min())
    rain_max_val = float(X['Rainfall_mm'].max())
    rain_range   = fc2.slider("Rainfall range (mm)", rain_min_val, rain_max_val, (rain_min_val, rain_max_val))

    fc3, fc4 = st.columns(2)
    temp_min_val = float(X['Temperature_C'].min())
    temp_max_val = float(X['Temperature_C'].max())
    temp_range   = fc3.slider("Temperature range (°C)", temp_min_val, temp_max_val, (temp_min_val, temp_max_val))

    sample_pct = fc4.slider("Use what % of filtered rows", 10, 100, 100)

    st.markdown("##### Step 2 — Choose a model")
    exp_model_name = st.selectbox("Model for this experiment",
                                   ["Logistic Regression", "Random Forest", "Gradient Boosting"])

    exp_test_size = st.slider("Experiment test split %", 10, 40, 20) / 100
    run_exp = st.button("▶️ Run Custom Experiment", type="primary")

    if run_exp:
        X_exp = X.copy()
        y_exp = y.copy()

        mask = (
            (X_exp['Month'] >= month_range[0]) & (X_exp['Month'] <= month_range[1]) &
            (X_exp['Rainfall_mm'] >= rain_range[0]) & (X_exp['Rainfall_mm'] <= rain_range[1]) &
            (X_exp['Temperature_C'] >= temp_range[0]) & (X_exp['Temperature_C'] <= temp_range[1])
        )
        X_exp = X_exp[mask]; y_exp = y_exp[mask]

        if sample_pct < 100:
            sample_n = max(int(len(X_exp) * sample_pct / 100), 10)
            idx = np.random.RandomState(42).choice(len(X_exp), sample_n, replace=False)
            X_exp = X_exp.iloc[idx]; y_exp = y_exp.iloc[idx]

        if len(X_exp) < 20:
            st.error("⚠️ Not enough rows after filtering (need at least 20). Please widen your filter ranges.")
        elif y_exp.nunique() < 2:
            st.error("⚠️ Filtered data contains only one class. Please adjust the filters.")
        else:
            st.info(
                f"🔬 Training on **{len(X_exp)} rows** "
                f"({int((y_exp==0).sum())} low-risk, {int((y_exp==1).sum())} high-risk) — "
                f"original dataset untouched ({len(X)} rows)."
            )

            Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(
                X_exp, y_exp, test_size=exp_test_size, random_state=42, stratify=y_exp
            )
            sc_e = StandardScaler()
            Xtr_e_sc = sc_e.fit_transform(Xtr_e)
            Xte_e_sc = sc_e.transform(Xte_e)

            if exp_model_name == "Logistic Regression":
                m_exp = LogisticRegression(max_iter=1000, random_state=42)
                m_exp.fit(Xtr_e_sc, ytr_e)
                yp_e = m_exp.predict(Xte_e_sc); ypr_e = m_exp.predict_proba(Xte_e_sc)[:, 1]
            elif exp_model_name == "Random Forest":
                m_exp = RandomForestClassifier(n_estimators=100, random_state=42)
                m_exp.fit(Xtr_e, ytr_e)
                yp_e = m_exp.predict(Xte_e); ypr_e = m_exp.predict_proba(Xte_e)[:, 1]
            else:
                m_exp = GradientBoostingClassifier(n_estimators=100, random_state=42)
                m_exp.fit(Xtr_e, ytr_e)
                yp_e = m_exp.predict(Xte_e); ypr_e = m_exp.predict_proba(Xte_e)[:, 1]

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
                delta_str = None
                if exp_model_name in results:
                    delta_val = val - results[exp_model_name][metric]
                    delta_str = f"{delta_val:+.4f} vs full data"
                col.metric(metric, f"{val:.4f}", delta_str)

            if exp_model_name in results:
                st.markdown("##### Confusion Matrix Comparison")
                cm_cols = st.columns(2)
                for idx_cm, (label_cm, cm_data) in enumerate([
                    (f"{exp_model_name} — Custom ({len(X_exp)} rows)", confusion_matrix(yte_e, yp_e)),
                    (f"{exp_model_name} — Full Data ({len(X)} rows)",  results[exp_model_name]['cm']),
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
                            ax.text(c, r, str(cm_data[r, c]), ha='center', va='center',
                                    fontsize=13, fontweight='bold',
                                    color='white' if cm_data[r,c] > thresh else 'black')
                    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                    ax.set_title(label_cm, fontweight='bold', fontsize=9)
                    plt.tight_layout()
                    cm_cols[idx_cm].pyplot(fig); plt.close()
            else:
                st.info("Train the main models first to see a side-by-side comparison.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Group 3 | BSc Data Science | Meru University of Science and Technology | 2026")
