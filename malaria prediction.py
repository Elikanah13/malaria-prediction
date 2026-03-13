# ================================================================
#  MALARIA HIGH-RISK PREDICTOR  |  Streamlit App  (Cloud-Ready)
#  - File upload instead of hardcoded path
#  - BytesIO for all model artifacts (no disk writes)
#  - Works on Streamlit Cloud, local, or any environment
# ================================================================

import io
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.metrics           import (accuracy_score, roc_auc_score, f1_score,
                                       precision_score, recall_score,
                                       confusion_matrix, classification_report,
                                       roc_curve)
import joblib

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Malaria Risk Predictor",
    page_icon  = "🦟",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #060d18;
    color: #dde6f0;
}

[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #112240;
}
[data-testid="stSidebar"] * { color: #8fa8c8 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e2eeff !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1f3c;
    border: 2px dashed #1e4080;
    border-radius: 12px;
    padding: 12px;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0a1628;
    border: 1px solid #1a3a6b;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #4a7ab5 !important;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #4fc3f7 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628;
    border-bottom: 1px solid #112240;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a7ab5;
    border: none;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 20px;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: #0d2044 !important;
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 28px;
    letter-spacing: 0.06em;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #0d2044 !important;
    border: 1px solid #1a4a8a !important;
    color: #4fc3f7 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
}

/* Headers */
h1 { font-family: 'Outfit', sans-serif !important; font-weight: 800 !important;
     color: #e8f4ff !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
          color: #a8c8e8 !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #1a3a6b; border-radius: 8px; }

/* Alerts */
.stSuccess { background: #022820 !important; border-left: 4px solid #10b981 !important; }
.stInfo    { background: #071d3a !important; border-left: 4px solid #4fc3f7 !important; }
.stWarning { background: #1c1206 !important; border-left: 4px solid #f59e0b !important; }
.stError   { background: #200a0a !important; border-left: 4px solid #ef4444 !important; }

/* Selectbox / slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #4a7ab5 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0a1628 !important;
    border: 1px solid #1a3a6b !important;
    border-radius: 8px !important;
    color: #8fa8c8 !important;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #6366f1) !important; }

code { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  CHART THEME HELPER
# ════════════════════════════════════════════════════════════════
PAL  = ["#4fc3f7", "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]
BG   = "#0a1628"
GRID = "#112240"
TXT  = "#8fa8c8"

def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    if title:
        ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    return ax

# ════════════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS  (pure — no disk I/O)
# ════════════════════════════════════════════════════════════════

def clean_data(df_raw: pd.DataFrame):
    """Step 2 – clean the raw dataframe."""
    df = df_raw.copy()

    # record before-state
    null_before = df.isnull().sum()
    dup_before  = df.duplicated().sum()

    # drop cols ≥ 40% null
    drop_cols = [c for c in df.columns if df[c].isnull().mean() >= 0.40]
    df.drop(columns=drop_cols, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # impute residual nulls
    for col in df.select_dtypes("number").columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes("object").columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df, drop_cols, null_before, int(dup_before)


def engineer_features(df: pd.DataFrame, target: str = "High_Risk_Binary"):
    """Step 3 – encode + derive features, return X, y, encoders, scores."""
    le_region = LabelEncoder()
    le_county = LabelEncoder()

    df = df.copy()
    df["Region_enc"]      = le_region.fit_transform(df["Region"])
    df["County_enc"]      = le_county.fit_transform(df["County"])
    df["Cases_per_Pop"]   = df["Malaria_Cases"] / df["Population"] * 1e5
    df["Lag_Change"]      = df["Malaria_Cases"] - df["Lag_1_Month_Cases"]
    df["Lag_Change_Pct"]  = df["Lag_Change"] / (df["Lag_1_Month_Cases"] + 1)
    df["Rain_x_Temp"]     = df["Rainfall_mm"] * df["Temperature_C"]
    df["Humidity_x_Rain"] = df["Humidity_percent"] * df["Rainfall_mm"]
    df["Season_enc"]      = LabelEncoder().fit_transform(
        df["Month"].map(lambda m: "LongRain"  if m in [3, 4, 5]    else
                                  "ShortRain" if m in [10, 11, 12]  else "Dry"))

    EXCLUDE      = ["Region", "County", target]
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols]
    y = df[target]

    # ANOVA feature selection
    K        = 10
    selector = SelectKBest(f_classif, k=K)
    selector.fit(X, y)
    scores   = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
    top_feats= scores.head(K).index.tolist()

    return X[top_feats], y, le_region, le_county, scores, top_feats, df


def split_scale(X, y, test_size=0.20):
    """Step 4 – stratified split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, scaler


def train_and_evaluate(X_tr_sc, X_te_sc, y_train, y_test):
    """Steps 5-7 – train 4 models, CV + hold-out evaluation."""
    MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       min_samples_leaf=2, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=5, random_state=42),
        "SVM (RBF)"          : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    prog = st.progress(0, text="Training models…")
    for i, (name, model) in enumerate(MODELS.items()):
        prog.progress((i + 1) / len(MODELS), text=f"Training {name}…")
        model.fit(X_tr_sc, y_train)
        cv_auc = cross_val_score(model, X_tr_sc, y_train, cv=skf, scoring="roc_auc")
        cv_acc = cross_val_score(model, X_tr_sc, y_train, cv=skf, scoring="accuracy")

        yp     = model.predict(X_te_sc)
        yproba = model.predict_proba(X_te_sc)[:, 1]
        results[name] = {
            "model"      : model,
            "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
            "cv_acc_mean": cv_acc.mean(),
            "y_pred"     : yp,
            "y_proba"    : yproba,
            "acc"  : accuracy_score(y_test, yp),
            "auc"  : roc_auc_score(y_test, yproba),
            "f1"   : f1_score(y_test, yp),
            "prec" : precision_score(y_test, yp),
            "rec"  : recall_score(y_test, yp),
            "cm"   : confusion_matrix(y_test, yp),
        }
    prog.empty()
    return results


def make_artifacts(best_model, scaler, le_region, le_county, top_feats):
    """Step 8 – dump artifacts to BytesIO buffers (no disk I/O)."""
    bufs = {}
    for key, obj in [("model", best_model), ("scaler", scaler),
                     ("le_region", le_region), ("le_county", le_county)]:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        bufs[key] = buf

    feat_csv = pd.Series(top_feats, name="feature").to_csv(index=False)
    return bufs, feat_csv


# ════════════════════════════════════════════════════════════════
#  SIDEBAR — UPLOAD + CONFIG
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦟 Malaria Predictor")
    st.markdown("---")

    # ── FILE UPLOAD (fixes the path error) ──────────────────────
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        label       = "Upload CSV file",
        type        = ["csv"],
        help        = "Upload Final_Malaria_Dataset.csv or any compatible malaria CSV",
        label_visibility = "collapsed",
    )

    # fallback: look for file next to this script
    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CSV  = os.path.join(BASE_DIR, "Final_Malaria_Dataset.csv")

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ {uploaded_file.name}  ({len(df_raw):,} rows)")
    elif os.path.exists(DEFAULT_CSV):
        df_raw = pd.read_csv(DEFAULT_CSV)
        st.info(f"📄 Using bundled CSV  ({len(df_raw):,} rows)")
    else:
        df_raw = None

    st.markdown("---")

    if df_raw is not None:
        st.markdown("### ⚙️ Pipeline Config")
        target_col = st.selectbox("🎯 Target Column", ["High_Risk_Binary"])
        test_split = st.slider("✂️ Test Split %", 10, 35, 20)
        st.markdown("---")
        run_btn = st.button("🚀  Run Full Pipeline")
    else:
        run_btn = False

    st.markdown("""
<small style='color:#2a4a6b;line-height:1.8'>
① Load Dataset<br>
② Data Cleaning<br>
③ Feature Engineering<br>
④ Feature Selection (ANOVA)<br>
⑤ Train / Test Split<br>
⑥ Train 4 Models<br>
⑦ Evaluate & Compare<br>
⑧ Deploy & Download
</small>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:24px 0 6px 0'>
  <span style='font-family:"IBM Plex Mono",monospace;font-size:11px;
               color:#4fc3f7;letter-spacing:0.2em'>ML PIPELINE  •  BINARY CLASSIFICATION</span>
  <h1 style='margin:4px 0 0 0;font-size:2.3rem'>
    🦟 Malaria High-Risk Predictor
  </h1>
  <p style='color:#2a5a8b;margin-top:6px;font-size:15px'>
    Upload dataset · Configure · Train · Evaluate · Deploy
  </p>
</div>
<hr style='border-color:#0d2044;margin-bottom:20px'>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  NO DATA STATE
# ════════════════════════════════════════════════════════════════
if df_raw is None:
    st.warning("👈 **Upload your CSV dataset** using the sidebar to get started.")
    with st.expander("📋 Expected CSV format", expanded=True):
        st.markdown("""
**Required columns** (at minimum):
```
Region, County, Year, Month, Population, Rainfall_mm,
Temperature_C, Humidity_percent, Malaria_Cases,
Lag_1_Month_Cases, Incidence_per_100k, High_Risk_Binary
```
**Optional** (will be dropped if >40% null):
```
ID, Health_Facilities, Avg_Income, Disease_Cases, Notes
```
        """)
    st.stop()

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Explorer",
    "🧹 Cleaning",
    "🧠 Training",
    "📈 Evaluation",
    "🔮 Predict",
    "💾 Export",
])

# ────────────────────────────────────────────────────────────────
#  TAB 1  –  DATA EXPLORER
# ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Step 1 · Raw Dataset")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",      f"{df_raw.shape[0]:,}")
    c2.metric("Columns",   df_raw.shape[1])
    c3.metric("Missing",   int(df_raw.isnull().sum().sum()))
    c4.metric("Duplicates",int(df_raw.duplicated().sum()))
    vc = df_raw["High_Risk_Binary"].value_counts()
    c5.metric("High Risk %", f"{vc.get(1,0)/len(df_raw)*100:.1f}%")

    with st.expander("🔍 Preview  (first 20 rows)", expanded=True):
        st.dataframe(df_raw.head(20), use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Missing Values per Column")
        null_df = df_raw.isnull().sum().reset_index()
        null_df.columns = ["Column", "Missing"]
        null_df["% Missing"] = (null_df["Missing"] / len(df_raw) * 100).round(1)
        null_df = null_df[null_df["Missing"] > 0]
        if null_df.empty:
            st.success("✅ No missing values found.")
        else:
            st.dataframe(null_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="#060d18")
        style_ax(ax)
        vc_vals = df_raw["High_Risk_Binary"].value_counts().sort_index()
        bars = ax.bar(["Low Risk (0)", "High Risk (1)"],
                      vc_vals.values,
                      color=["#34d399", "#fb7185"], edgecolor=GRID, width=0.5)
        for b, v in zip(bars, vc_vals.values):
            ax.text(b.get_x() + b.get_width()/2, v + 4, str(v),
                    ha="center", color=TXT, fontweight="bold", fontsize=11)
        ax.set_ylabel("Count", color=TXT)
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

# ────────────────────────────────────────────────────────────────
#  TAB 2  –  CLEANING PREVIEW
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Step 2 · Data Cleaning")

    df_clean, dropped_cols, null_before, dups_removed = clean_data(df_raw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cols Dropped",    len(dropped_cols))
    c2.metric("Duplicates Removed", dups_removed)
    c3.metric("Nulls After",     int(df_clean.isnull().sum().sum()))
    c4.metric("Clean Shape",     f"{df_clean.shape[0]} × {df_clean.shape[1]}")

    if dropped_cols:
        st.warning(f"🗑️  **Dropped {len(dropped_cols)} columns** with ≥ 40% missing values:  "
                   f"`{'`,  `'.join(dropped_cols)}`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Before Cleaning**")
        nb = null_before.reset_index()
        nb.columns = ["Column","Missing"]
        nb["% Missing"] = (nb["Missing"] / len(df_raw) * 100).round(1)
        st.dataframe(nb, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**After Cleaning  (remaining nulls)**")
        na = df_clean.isnull().sum().reset_index()
        na.columns = ["Column","Missing"]
        na = na[na["Missing"] > 0]
        if na.empty:
            st.success("✅ Zero missing values remaining.")
        else:
            st.dataframe(na, use_container_width=True, hide_index=True)

    with st.expander("🔍 Cleaned Data Preview"):
        st.dataframe(df_clean.head(20), use_container_width=True)

# ────────────────────────────────────────────────────────────────
#  TAB 3  –  TRAINING
# ────────────────────────────────────────────────────────────────
with tab3:
    if not run_btn and "pipeline" not in st.session_state:
        st.info("⬅️  Configure the pipeline in the sidebar and click **🚀 Run Full Pipeline**.")
        st.stop()

    # ── Run pipeline ─────────────────────────────────────────────
    if run_btn:
        with st.status("⚙️  Running ML Pipeline…", expanded=True) as status:

            st.write("🧹  Step 2 · Cleaning data…")
            df_clean, dropped_cols, null_before, dups_removed = clean_data(df_raw)
            st.write(f"   Dropped {len(dropped_cols)} columns · {dups_removed} duplicates removed")

            st.write("⚙️  Step 3 · Engineering & selecting features…")
            X_sel, y, le_region, le_county, scores_s, top_feats, df_eng = \
                engineer_features(df_clean, target=target_col)
            st.write(f"   {len(top_feats)} top features selected via ANOVA F-score")

            st.write(f"✂️  Step 4 · Splitting {100-test_split}/{test_split}…")
            (X_train, X_test, y_train, y_test,
             X_tr_sc, X_te_sc, scaler) = split_scale(X_sel, y, test_size=test_split/100)
            st.write(f"   Train={len(X_train)} · Test={len(X_test)}")

            st.write("🤖  Steps 5-7 · Training & evaluating 4 models…")
            results = train_and_evaluate(X_tr_sc, X_te_sc, y_train, y_test)

            best_name = max(results, key=lambda k: results[k]["auc"])

            st.write("💾  Step 8 · Packaging artifacts…")
            bufs, feat_csv = make_artifacts(
                results[best_name]["model"], scaler,
                le_region, le_county, top_feats)

            # store everything in session state
            st.session_state.pipeline = {
                "results"     : results,
                "best_name"   : best_name,
                "scores_s"    : scores_s,
                "top_feats"   : top_feats,
                "X_sel"       : X_sel,
                "X_train"     : X_train,
                "X_test"      : X_test,
                "y_train"     : y_train,
                "y_test"      : y_test,
                "X_tr_sc"     : X_tr_sc,
                "X_te_sc"     : X_te_sc,
                "scaler"      : scaler,
                "le_region"   : le_region,
                "le_county"   : le_county,
                "df_clean"    : df_clean,
                "df_eng"      : df_eng,
                "y"           : y,
                "bufs"        : bufs,
                "feat_csv"    : feat_csv,
                "dropped_cols": dropped_cols,
            }
            status.update(label="✅  Pipeline complete!", state="complete")

    # ── Display training results ──────────────────────────────────
    if "pipeline" not in st.session_state:
        st.stop()

    P = st.session_state.pipeline
    results   = P["results"]
    best_name = P["best_name"]
    scores_s  = P["scores_s"]
    top_feats = P["top_feats"]

    st.markdown("### Step 6 · Training Results  (5-Fold CV)")
    cv_cols = st.columns(4)
    SHORT   = ["LR", "RF", "GBM", "SVM"]
    for i, (name, short) in enumerate(zip(results, SHORT)):
        cv_cols[i].metric(
            label = short,
            value = f"{results[name]['cv_auc_mean']:.4f}",
            delta = f"AUC ± {results[name]['cv_auc_std']:.4f}",
        )

    st.markdown("### Top 10 Selected Features  (ANOVA F-Score)")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#060d18")
    style_ax(ax)
    top = scores_s.head(10).sort_values()
    colors = [PAL[i % len(PAL)] for i in range(len(top))]
    ax.barh(top.index, top.values, color=colors, edgecolor=GRID, height=0.65)
    for i, (v, _) in enumerate(zip(top.values, top.index)):
        ax.text(v + 10, i, f"{v:.0f}", va="center", color=TXT, fontsize=9,
                fontfamily="monospace")
    ax.set_xlabel("F-Score", color=TXT)
    ax.set_facecolor(BG)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Feature importance (Random Forest)
    if "Random Forest" in results:
        st.markdown("### Feature Importances  (Random Forest)")
        rf  = results["Random Forest"]["model"]
        fi  = pd.Series(rf.feature_importances_, index=top_feats).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#060d18")
        style_ax(ax)
        ax.barh(fi.index, fi.values,
                color=[PAL[i % len(PAL)] for i in range(len(fi))], edgecolor=GRID)
        ax.set_xlabel("Importance", color=TXT)
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ────────────────────────────────────────────────────────────────
#  TAB 4  –  EVALUATION
# ────────────────────────────────────────────────────────────────
with tab4:
    if "pipeline" not in st.session_state:
        st.info("⬅️  Run the pipeline first.")
        st.stop()

    P         = st.session_state.pipeline
    results   = P["results"]
    best_name = P["best_name"]
    y_test    = P["y_test"]
    SHORT     = ["LR", "RF", "GBM", "SVM"]

    st.markdown("### Step 7 · Model Comparison")

    # summary table
    summary = pd.DataFrame([{
        "Model"    : n,
        "CV AUC"   : f"{results[n]['cv_auc_mean']:.4f}",
        "Test Acc" : f"{results[n]['acc']:.4f}",
        "AUC-ROC"  : f"{results[n]['auc']:.4f}",
        "F1"       : f"{results[n]['f1']:.4f}",
        "Precision": f"{results[n]['prec']:.4f}",
        "Recall"   : f"{results[n]['rec']:.4f}",
        "Best ?"   : "🏆" if n == best_name else "",
    } for n in results])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Charts row 1 ────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**ROC Curves**")
        fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="#060d18")
        style_ax(ax)
        for i, (name, short) in enumerate(zip(results, SHORT)):
            fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
            ax.plot(fpr, tpr, color=PAL[i], lw=2,
                    label=f"{short}  {results[name]['auc']:.3f}")
        ax.plot([0,1],[0,1], "--", color=GRID, lw=1)
        ax.set_xlabel("False Positive Rate", color=TXT)
        ax.set_ylabel("True Positive Rate",  color=TXT)
        ax.legend(labelcolor=TXT, facecolor=BG, fontsize=9, framealpha=0.4)
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("**CV AUC by Model**")
        fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="#060d18")
        style_ax(ax)
        cvm = [results[n]["cv_auc_mean"] for n in results]
        cvs = [results[n]["cv_auc_std"]  for n in results]
        bars = ax.bar(SHORT, cvm, yerr=cvs, color=PAL[:4], capsize=5,
                      edgecolor=GRID, width=0.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("AUC", color=TXT)
        for b, v in zip(bars, cvm):
            ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}",
                    ha="center", color=TXT, fontsize=9, fontweight="bold")
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Per-model detail ─────────────────────────────────────────
    st.markdown("### Inspect a Model")
    chosen = st.selectbox("Select model:", list(results.keys()))
    info   = results[chosen]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Confusion Matrix · {chosen}**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#060d18")
        style_ax(ax)
        cm = info["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low Risk", "High Risk"], color=TXT)
        ax.set_yticklabels(["Low Risk", "High Risk"], color=TXT)
        ax.set_xlabel("Predicted", color=TXT)
        ax.set_ylabel("Actual",    color=TXT)
        tn, fp, fn, tp = cm.ravel()
        for (r, c), val in [((0,0),tn), ((0,1),fp), ((1,0),fn), ((1,1),tp)]:
            ax.text(c, r, val, ha="center", va="center",
                    color="white", fontsize=20, fontweight="bold")
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**Classification Report**")
        rpt = classification_report(
            y_test, info["y_pred"],
            target_names=["Low Risk", "High Risk"],
            output_dict=True
        )
        rpt_df = pd.DataFrame(rpt).T.round(4)
        st.dataframe(rpt_df, use_container_width=True)

    # ── Test metrics bar chart ───────────────────────────────────
    st.markdown("### All Metrics Compared")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#060d18")
    style_ax(ax)
    mets = ["acc","auc","f1","prec","rec"]
    labs = ["Accuracy","AUC","F1","Precision","Recall"]
    xb   = np.arange(len(mets)); w = 0.18
    for i, (name, short) in enumerate(zip(results, SHORT)):
        ax.bar(xb+i*w, [results[name][m] for m in mets], w,
               label=short, color=PAL[i], edgecolor=GRID, alpha=0.88)
    ax.set_xticks(xb+w*1.5)
    ax.set_xticklabels(labs, color=TXT, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color=TXT)
    ax.legend(labelcolor=TXT, facecolor=BG, fontsize=9, ncol=4, framealpha=0.4)
    ax.set_facecolor(BG)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ────────────────────────────────────────────────────────────────
#  TAB 5  –  PREDICT
# ────────────────────────────────────────────────────────────────
with tab5:
    if "pipeline" not in st.session_state:
        st.info("⬅️  Run the pipeline first.")
        st.stop()

    P         = st.session_state.pipeline
    best_name = P["best_name"]
    best_model= P["results"][best_name]["model"]
    scaler    = P["scaler"]
    top_feats = P["top_feats"]
    X_sel     = P["X_sel"]

    st.markdown(f"### Step 8 · Predict with **{best_name}**")
    st.caption("Adjust the feature values below, then click **Predict**.")

    last_row = X_sel.iloc[-1]
    inputs   = {}
    pairs    = [st.columns(2) for _ in range((len(top_feats) + 1) // 2)]

    for i, feat in enumerate(top_feats):
        col = pairs[i // 2][i % 2]
        mn  = float(X_sel[feat].min())
        mx  = float(X_sel[feat].max())
        stp = max((mx - mn) / 200, 0.0001)
        inputs[feat] = col.number_input(
            feat, value=float(last_row[feat]),
            min_value=mn - abs(mn), max_value=mx + abs(mx),
            step=stp, format="%.4f"
        )

    st.markdown("---")
    if st.button("🔮  Predict Risk Level"):
        row_df = pd.DataFrame([inputs])[top_feats]
        row_sc = scaler.transform(row_df)
        pred   = best_model.predict(row_sc)[0]
        proba  = best_model.predict_proba(row_sc)[0]

        if pred == 1:
            st.error(f"## 🔴 HIGH RISK  ·  Confidence: {max(proba)*100:.1f}%")
        else:
            st.success(f"## 🟢 LOW RISK  ·  Confidence: {max(proba)*100:.1f}%")

        m1, m2 = st.columns(2)
        m1.metric("Probability  HIGH RISK", f"{proba[1]*100:.1f}%")
        m2.metric("Probability  LOW RISK",  f"{proba[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(7, 1.8), facecolor="#060d18")
        style_ax(ax)
        ax.barh(["High Risk ▲"], [proba[1]], color="#fb7185", height=0.4)
        ax.barh(["Low Risk ▼"],  [proba[0]], color="#34d399", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Probabilities", color=TXT, fontsize=10)
        for s in ["top","right","left"]: ax.spines[s].set_visible(False)
        ax.set_facecolor(BG)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Batch prediction on recent rows ────────────────────────
    st.markdown("---")
    st.markdown("#### Batch Prediction on Historical Data")
    n_rows   = st.slider("Show last N rows", 5, 60, 20)
    X_sample = scaler.transform(X_sel.iloc[-n_rows:][top_feats])
    preds_h  = best_model.predict(X_sample)
    proba_h  = best_model.predict_proba(X_sample)[:, 1]
    batch_df = X_sel.iloc[-n_rows:][top_feats].copy()
    batch_df["Predicted"]  = ["🔴 HIGH" if p == 1 else "🟢 LOW" for p in preds_h]
    batch_df["Prob High %"]= (proba_h * 100).round(1)
    batch_df["Actual"]     = P["y"].iloc[-n_rows:].map({1:"🔴 HIGH", 0:"🟢 LOW"}).values
    batch_df["✓"]          = (batch_df["Predicted"] == batch_df["Actual"]).map(
                               {True:"✅", False:"❌"})
    st.dataframe(
        batch_df[["Predicted","Prob High %","Actual","✓"]],
        use_container_width=True
    )
    correct = (batch_df["✓"] == "✅").sum()
    st.caption(f"Correct: {correct}/{n_rows}  ({correct/n_rows*100:.0f}%)")

# ────────────────────────────────────────────────────────────────
#  TAB 6  –  EXPORT
# ────────────────────────────────────────────────────────────────
with tab6:
    if "pipeline" not in st.session_state:
        st.info("⬅️  Run the pipeline first to unlock exports.")
        st.stop()

    P         = st.session_state.pipeline
    best_name = P["best_name"]
    results   = P["results"]
    bufs      = P["bufs"]
    feat_csv  = P["feat_csv"]
    y_test    = P["y_test"]

    st.markdown("### 💾 Download Artifacts")
    st.caption("All artifacts are generated in-memory — no disk writes required.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"**🧠 Best Model**")
        st.caption(f"{best_name}")
        st.download_button("⬇️ model.pkl",   data=bufs["model"],
                           file_name="malaria_model.pkl",
                           mime="application/octet-stream")

    with c2:
        st.markdown("**⚖️ Scaler**")
        st.caption("StandardScaler")
        st.download_button("⬇️ scaler.pkl",  data=bufs["scaler"],
                           file_name="malaria_scaler.pkl",
                           mime="application/octet-stream")

    with c3:
        st.markdown("**📋 Features**")
        st.caption("Selected feature list")
        st.download_button("⬇️ features.csv", data=feat_csv,
                           file_name="malaria_features.csv",
                           mime="text/csv")

    with c4:
        st.markdown("**📊 Results**")
        st.caption("Evaluation metrics")
        res_csv = pd.DataFrame([{
            "Model"    : n,
            "CV_AUC"   : round(results[n]["cv_auc_mean"], 4),
            "Test_Acc" : round(results[n]["acc"],  4),
            "AUC_ROC"  : round(results[n]["auc"],  4),
            "F1"       : round(results[n]["f1"],   4),
            "Precision": round(results[n]["prec"], 4),
            "Recall"   : round(results[n]["rec"],  4),
        } for n in results]).to_csv(index=False)
        st.download_button("⬇️ results.csv",  data=res_csv,
                           file_name="model_results.csv",
                           mime="text/csv")

    st.markdown("---")
    st.markdown("### 🔌 Inference Code  (copy for production)")
    st.code(f'''
import joblib, pandas as pd

# Load saved artifacts
model    = joblib.load("malaria_model.pkl")
scaler   = joblib.load("malaria_scaler.pkl")
features = pd.read_csv("malaria_features.csv")["feature"].tolist()

def predict_risk(data: dict) -> dict:
    """
    data : dict with keys matching features list
    Returns risk level, confidence, and probabilities
    """
    df     = pd.DataFrame([data])[features]
    scaled = scaler.transform(df)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]
    return {{
        "risk_level" : "HIGH RISK" if pred == 1 else "LOW RISK",
        "confidence" : f"{{max(proba)*100:.1f}}%",
        "prob_high"  : round(float(proba[1]), 4),
        "prob_low"   : round(float(proba[0]), 4),
    }}

# Example
result = predict_risk({{
    "Incidence_per_100k" : 145.3,
    "Cases_per_Pop"      : 145.3,
    "Malaria_Cases"      : 1050,
    "Lag_1_Month_Cases"  : 980,
    "Population"         : 750000,
    "Lag_Change"         : 70,
    "Rainfall_mm"        : 120.5,
    "Humidity_x_Rain"    : 8472.5,
    "Rain_x_Temp"        : 2651.0,
    "County_enc"         : 3,
}})
print(result)
# Output: {{"risk_level": "HIGH RISK", "confidence": "97.0%", ...}}
''', language="python")

    st.markdown("---")
    st.markdown("### 📦 requirements.txt")
    st.code("""streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
joblib>=1.3.0""", language="text")
