# ================================================================
#  MALARIA HIGH-RISK PREDICTOR — Full Streamlit ML App
#  Target : High_Risk_Binary  (1 = High Risk | 0 = Low Risk)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, joblib, io
warnings.filterwarnings("ignore")

from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.metrics           import (accuracy_score, f1_score, precision_score,
                                       recall_score, roc_auc_score, roc_curve,
                                       confusion_matrix, classification_report)

PAL  = ["#00d4ff", "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]
BG   = "#0d1726"
GRID = "#1e2d45"
TEXT = "#cbd5e1"

def apply_dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Risk Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family:'Syne',sans-serif; background-color:#080c14; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0d1220 !important; border-right:1px solid #1e2d45; }
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
div[data-testid="metric-container"] { background:#0d1726; border:1px solid #1e3a5f; border-radius:12px; padding:16px 20px; }
div[data-testid="metric-container"] label { color:#64748b !important; font-size:12px; letter-spacing:0.1em; text-transform:uppercase; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#38bdf8 !important; font-family:'Space Mono',monospace; font-size:1.5rem !important; }
.stTabs [data-baseweb="tab-list"] { background:#0d1220; border-bottom:1px solid #1e2d45; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#475569; border:none; font-weight:600; padding:10px 22px; }
.stTabs [aria-selected="true"] { background:#0f2744 !important; color:#38bdf8 !important; border-bottom:2px solid #38bdf8 !important; }
.stButton > button { background:linear-gradient(135deg,#0ea5e9,#6366f1); color:white; border:none; border-radius:8px; font-family:'Space Mono',monospace; font-size:13px; padding:10px 28px; width:100%; transition:opacity 0.2s; }
.stButton > button:hover { opacity:0.85; }
h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; color:#f0f9ff !important; }
h2,h3 { font-family:'Syne',sans-serif !important; font-weight:600 !important; color:#cbd5e1 !important; }
.stSuccess { background:#042f2e !important; border-left:4px solid #10b981 !important; }
.stInfo    { background:#0c1a2e !important; border-left:4px solid #38bdf8 !important; }
.stError   { background:#2d0a0a !important; border-left:4px solid #fb7185 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Malaria Predictor")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.markdown("**⚙️ Pipeline Settings**")
    test_size = st.slider("Test Split %", 10, 40, 20)
    top_k     = st.slider("Features to Select (K)", 5, 16, 12)
    st.markdown("---")
    run_btn = st.button("🚀 Run Full Pipeline")
    st.markdown("---")
    st.markdown("""<small style='color:#334155'>
<b>Pipeline Steps</b><br>
① Load Dataset<br>② Data Cleaning<br>③ Feature Engineering<br>
④ Feature Selection<br>⑤ Train/Test Split<br>⑥ Train 4 Models<br>
⑦ Evaluate & Compare<br>⑧ Deploy & Predict</small>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='padding:28px 0 8px 0'>
  <span style='font-family:Space Mono,monospace;font-size:11px;color:#38bdf8;letter-spacing:0.2em'>MACHINE LEARNING PIPELINE</span>
  <h1 style='margin:4px 0 0 0;font-size:2.4rem'>🦟 Malaria High-Risk Predictor</h1>
  <p style='color:#475569;margin-top:6px'>Upload your dataset · Clean · Engineer · Train · Predict</p>
</div>
<hr style='border-color:#1e2d45;margin-bottom:24px'>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 Upload your CSV dataset in the sidebar to get started.")
    st.markdown("""
**Expected CSV columns:**
```
Region, County, Year, Month, Population, Rainfall_mm, Temperature_C,
Humidity_percent, Malaria_Cases, Lag_1_Month_Cases, Incidence_per_100k, High_Risk_Binary
```
""")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "🧠 Model Training", "📈 Evaluation", "🔮 Predict", "💾 Export"
])

# ================================================================
# TAB 1 — DATA EXPLORER
# ================================================================
with tab1:
    st.markdown("### Step 1 · Raw Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     df_raw.shape[0])
    c2.metric("Columns",        df_raw.shape[1])
    c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))

    with st.expander("🔍 Preview Data", expanded=True):
        st.dataframe(df_raw.head(20), use_container_width=True)

    if "High_Risk_Binary" in df_raw.columns:
        vc = df_raw["High_Risk_Binary"].value_counts()
        st.markdown("### Target Distribution")
        col_a, col_b = st.columns(2)
        col_a.metric("🟢 Low Risk  (0)", f"{vc.get(0,0)} rows ({vc.get(0,0)/len(df_raw)*100:.1f}%)")
        col_b.metric("🔴 High Risk (1)", f"{vc.get(1,0)} rows ({vc.get(1,0)/len(df_raw)*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="#080c14")
        ax = axes[0]; apply_dark(ax)
        ax.bar(["Low Risk (0)","High Risk (1)"], [vc.get(0,0),vc.get(1,0)],
               color=["#34d399","#fb7185"], edgecolor=GRID, width=0.5)
        ax.set_title("Class Count", color=TEXT, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", color=TEXT)
        for i, v in enumerate([vc.get(0,0), vc.get(1,0)]):
            ax.text(i, v+5, str(v), ha="center", color=TEXT, fontweight="bold")
        ax2 = axes[1]; apply_dark(ax2)
        if "Incidence_per_100k" in df_raw.columns:
            low_i  = df_raw[df_raw["High_Risk_Binary"]==0]["Incidence_per_100k"].dropna()
            high_i = df_raw[df_raw["High_Risk_Binary"]==1]["Incidence_per_100k"].dropna()
            ax2.hist(low_i,  bins=40, color="#34d399", alpha=0.7, label="Low Risk",  edgecolor=GRID)
            ax2.hist(high_i, bins=40, color="#fb7185", alpha=0.7, label="High Risk", edgecolor=GRID)
            ax2.set_title("Incidence per 100k", color=TEXT, fontsize=11, fontweight="bold")
            ax2.set_xlabel("Incidence per 100k"); ax2.legend(labelcolor=TEXT, facecolor=BG, fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("### Missing Values per Column")
    miss = df_raw.isnull().sum()
    miss_df = pd.DataFrame({"Column":miss.index,"Missing":miss.values,"% Missing":(miss.values/len(df_raw)*100).round(1)})
    st.dataframe(miss_df[miss_df["Missing"]>0], use_container_width=True, hide_index=True)
    st.markdown("### Descriptive Statistics")
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

# ================================================================
# TAB 2 — MODEL TRAINING
# ================================================================
with tab2:
    if not run_btn and "results" not in st.session_state:
        st.info("⬅️ Configure settings in the sidebar and click **🚀 Run Full Pipeline**.")
        st.stop()

    if run_btn:
        with st.status("⚙️ Running Full ML Pipeline…", expanded=True) as status:
            # STEP 2: Clean
            st.write("🧹 Step 2 · Cleaning data…")
            df = df_raw.copy()
            drop_cols = [c for c in df.columns if df[c].isnull().mean() >= 0.40]
            df.drop(columns=drop_cols, inplace=True)
            n_dup = df.duplicated().sum()
            df.drop_duplicates(inplace=True)
            for col in df.select_dtypes(include="number").columns:
                if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include="object").columns:
                if df[col].isnull().any(): df[col].fillna(df[col].mode()[0], inplace=True)
            st.write(f"   Dropped {len(drop_cols)} high-null columns, {n_dup} duplicate rows removed")

            # STEP 3: Feature engineering
            st.write("⚙️ Step 3 · Engineering features…")
            le_region = LabelEncoder(); le_county = LabelEncoder()
            df["Region_enc"]   = le_region.fit_transform(df["Region"])
            df["County_enc"]   = le_county.fit_transform(df["County"])
            df["Season"]       = df["Month"].map({1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:3,11:3,12:3})
            df["Cases_per_Pop"]   = df["Malaria_Cases"] / df["Population"] * 1e5
            df["Lag_Change"]      = df["Malaria_Cases"] - df["Lag_1_Month_Cases"]
            df["Lag_Ratio"]       = (df["Malaria_Cases"] / df["Lag_1_Month_Cases"].replace(0, np.nan)).fillna(1)
            df["Rain_x_Humidity"] = df["Rainfall_mm"]   * df["Humidity_percent"]
            df["Temp_x_Humidity"] = df["Temperature_C"] * df["Humidity_percent"]
            df["Rain_x_Temp"]     = df["Rainfall_mm"]   * df["Temperature_C"]
            st.write("   Created 7 new interaction + seasonal features")

            # STEP 4: Feature selection
            st.write(f"✅ Step 4 · Selecting top {top_k} features…")
            TARGET    = "High_Risk_Binary"
            feat_cols = [c for c in df.columns if c not in ["Region","County",TARGET]]
            X = df[feat_cols]; y = df[TARGET]
            selector  = SelectKBest(f_classif, k=min(top_k, len(feat_cols)))
            selector.fit(X, y)
            f_scores  = pd.Series(selector.scores_, index=feat_cols).sort_values(ascending=False)
            sel_feats = f_scores.head(top_k).index.tolist()
            X_sel     = X[sel_feats]

            # STEP 5: Split
            st.write(f"✂️ Step 5 · Splitting {100-test_size}/{test_size}…")
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=test_size/100, random_state=42, stratify=y)
            scaler     = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            # STEP 6: Train
            st.write("🤖 Step 6 · Training 4 models…")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            MODELS = {
                "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
                "SVM (RBF)":           SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
            }
            results = {}
            for name, model in MODELS.items():
                model.fit(X_train_sc, y_train)
                cv = cross_val_score(model, X_train_sc, y_train, cv=skf, scoring="roc_auc")
                results[name] = dict(model=model, cv_mean=cv.mean(), cv_std=cv.std())
                st.write(f"   ✓ {name}  —  CV AUC: {cv.mean():.4f}")

            # STEP 7: Evaluate
            st.write("📊 Step 7 · Evaluating on test set…")
            for name, info in results.items():
                m = info["model"]
                y_pred = m.predict(X_test_sc)
                y_proba = m.predict_proba(X_test_sc)[:,1]
                info.update(dict(
                    y_pred=y_pred, y_proba=y_proba,
                    acc=accuracy_score(y_test,y_pred),
                    auc=roc_auc_score(y_test,y_proba),
                    f1=f1_score(y_test,y_pred),
                    prec=precision_score(y_test,y_pred),
                    rec=recall_score(y_test,y_pred),
                    cm=confusion_matrix(y_test,y_pred),
                    report=classification_report(y_test,y_pred,target_names=["Low Risk","High Risk"],output_dict=True),
                ))

            best_name  = max(results, key=lambda k: results[k]["auc"])
            best_model = results[best_name]["model"]
            buf_model  = io.BytesIO(); joblib.dump(best_model, buf_model); buf_model.seek(0)
            buf_scaler = io.BytesIO(); joblib.dump(scaler,     buf_scaler); buf_scaler.seek(0)

            st.session_state.update(dict(
                results=results, best_name=best_name, best_model=best_model,
                scaler=scaler, sel_feats=sel_feats, f_scores=f_scores,
                X_sel=X_sel, X_test=X_test, y_test=y_test, y_train=y_train,
                df=df, drop_cols=drop_cols, n_dup=n_dup,
                buf_model=buf_model, buf_scaler=buf_scaler, top_k=top_k,
            ))
            status.update(label="✅ Pipeline complete!", state="complete")

    if "results" not in st.session_state:
        st.stop()

    results   = st.session_state.results
    best_name = st.session_state.best_name
    sel_feats = st.session_state.sel_feats
    f_scores  = st.session_state.f_scores
    top_k     = st.session_state.top_k
    y_train   = st.session_state.y_train
    y_test    = st.session_state.y_test

    st.markdown("### ✅ Step 2 · Data Cleaning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Columns Dropped",   len(st.session_state.drop_cols))
    c2.metric("Duplicates Removed", st.session_state.n_dup)
    c3.metric("Remaining Nulls",   0)
    if st.session_state.drop_cols:
        st.caption(f"Dropped: `{'`, `'.join(st.session_state.drop_cols)}`")

    st.markdown("### ✅ Step 3 & 4 · Feature Engineering & Selection")
    col_l, col_r = st.columns(2)
    with col_l:
        feat_df = pd.DataFrame({"Feature":f_scores.head(top_k).index,"F-Score":f_scores.head(top_k).values.round(2)})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    with col_r:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#080c14"); apply_dark(ax)
        fs = f_scores.head(top_k).sort_values()
        ax.barh(fs.index, fs.values, color=[PAL[i%len(PAL)] for i in range(len(fs))], edgecolor=GRID, height=0.65)
        ax.set_xlabel("F-Score"); ax.set_title("ANOVA F-Scores", color=TEXT, fontsize=11, fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("### ✅ Step 5 · Train / Test Split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Samples",   len(y_train))
    c2.metric("Test Samples",    len(y_test))
    c3.metric("Train High Risk", int(y_train.sum()))
    c4.metric("Test High Risk",  int(y_test.sum()))

    st.markdown("### ✅ Step 6 · Cross-Validation AUC")
    cv_df = pd.DataFrame([{
        "Model":       n,
        "CV AUC Mean": f"{i['cv_mean']:.4f}",
        "CV AUC Std":  f"±{i['cv_std']:.4f}",
        "Best?":       "🏆" if n == best_name else ""
    } for n, i in results.items()])
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

# ================================================================
# TAB 3 — EVALUATION
# ================================================================
with tab3:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first."); st.stop()

    results   = st.session_state.results
    best_name = st.session_state.best_name
    y_test    = st.session_state.y_test
    short     = ["LR","RF","GBM","SVM"]

    st.markdown("### Step 7 · Test Set Results")
    summary = pd.DataFrame([{
        "Model":     n,
        "Accuracy":  f"{i['acc']:.4f}",
        "AUC-ROC":   f"{i['auc']:.4f}",
        "F1 Score":  f"{i['f1']:.4f}",
        "Precision": f"{i['prec']:.4f}",
        "Recall":    f"{i['rec']:.4f}",
        "Best?":     "🏆" if n == best_name else ""
    } for n, i in results.items()])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ROC Curves**")
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#080c14"); apply_dark(ax)
        for i, (name, info) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, info["y_proba"])
            ax.plot(fpr, tpr, color=PAL[i], lw=2, label=f"{short[i]}  AUC={info['auc']:.3f}")
        ax.plot([0,1],[0,1],"--",color=GRID,lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", color=TEXT, fontsize=12, fontweight="bold")
        ax.legend(labelcolor=TEXT, facecolor=BG, fontsize=9, framealpha=0.4)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown(f"**Confusion Matrix — {best_name}**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#080c14"); apply_dark(ax)
        cm = results[best_name]["cm"]
        ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white", fontsize=20, fontweight="bold")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Low Risk","High Risk"], color=TEXT)
        ax.set_yticklabels(["Low Risk","High Risk"], color=TEXT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix ({best_name})", color=TEXT, fontsize=11, fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**All Metrics Comparison**")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#080c14"); apply_dark(ax)
    mets = ["acc","auc","f1","prec","rec"]; xlbls = ["Accuracy","AUC","F1","Precision","Recall"]
    x, w = np.arange(len(mets)), 0.18
    for i, (name, info) in enumerate(results.items()):
        ax.bar(x+i*w, [info[m] for m in mets], w, label=short[i], color=PAL[i], edgecolor=GRID, alpha=0.88)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(xlbls, color=TEXT, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(labelcolor=TEXT, facecolor=BG, fontsize=9, framealpha=0.4)
    ax.set_title("Test Metrics — All Models", color=TEXT, fontsize=12, fontweight="bold")
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("**Detailed Classification Reports**")
    tabs_models = st.tabs(list(results.keys()))
    for i, (name, info) in enumerate(results.items()):
        with tabs_models[i]:
            st.dataframe(pd.DataFrame(info["report"]).T.round(3), use_container_width=True)

    st.markdown("**Monthly Average Cases Trend**")
    df_sess = st.session_state.df
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#080c14"); apply_dark(ax)
    mo = df_sess.groupby("Month")["Malaria_Cases"].mean()
    ax.fill_between(mo.index, mo.values, alpha=0.15, color=PAL[0])
    ax.plot(mo.index, mo.values, color=PAL[0], lw=2.5, marker="o", markersize=6)
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Cases")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], color=TEXT, fontsize=9)
    ax.set_title("Monthly Average Malaria Cases", color=TEXT, fontsize=12, fontweight="bold")
    for start, end in [(3,6),(10,13)]:
        ax.axvspan(start, min(end,12.5), alpha=0.07, color="#34d399")
    st.pyplot(fig, use_container_width=True); plt.close()

# ================================================================
# TAB 4 — PREDICT
# ================================================================
with tab4:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first."); st.stop()

    best_model = st.session_state.best_model
    scaler     = st.session_state.scaler
    sel_feats  = st.session_state.sel_feats
    X_sel      = st.session_state.X_sel
    best_name  = st.session_state.best_name
    X_test     = st.session_state.X_test
    y_test     = st.session_state.y_test

    st.markdown(f"### Step 8 · Predict with **{best_name}**")
    st.markdown("Adjust the values below and click **Predict**.")

    last_row = X_sel.iloc[-1]
    inputs   = {}
    n_cols   = 3
    rows_    = [st.columns(n_cols) for _ in range((len(sel_feats)+n_cols-1)//n_cols)]
    for i, feat in enumerate(sel_feats):
        col  = rows_[i//n_cols][i%n_cols]
        mn   = float(X_sel[feat].min())
        mx   = float(X_sel[feat].max())
        step = max((mx-mn)/200, 0.001)
        inputs[feat] = col.number_input(feat, value=float(last_row[feat]),
                                         min_value=mn-abs(mn), max_value=mx+abs(mx),
                                         step=step, format="%.4f")

    if st.button("🔮 Predict Risk Level"):
        row_df = pd.DataFrame([inputs])[sel_feats]
        row_sc = scaler.transform(row_df)
        pred   = best_model.predict(row_sc)[0]
        proba  = best_model.predict_proba(row_sc)[0]
        conf   = max(proba) * 100
        if pred == 1:
            st.error(  f"## 🔴 HIGH RISK  —  Confidence: {conf:.1f}%")
        else:
            st.success(f"## 🟢 LOW RISK   —  Confidence: {conf:.1f}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob High Risk", f"{proba[1]*100:.1f}%")
        c2.metric("Prob Low Risk",  f"{proba[0]*100:.1f}%")
        c3.metric("Model Used",     best_name[:18])
        fig, ax = plt.subplots(figsize=(7, 1.8), facecolor="#080c14"); apply_dark(ax)
        ax.barh(["Low Risk"],  [proba[0]], color="#34d399", height=0.4)
        ax.barh(["High Risk"], [proba[1]], color="#fb7185", height=0.4)
        ax.set_xlim(0,1); ax.set_title("Prediction Probabilities", color=TEXT, fontsize=10)
        for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("#### Batch Predictions on Recent Test Rows")
    n_rows    = st.slider("Number of rows to show", 5, 50, 15)
    X_samp_sc = scaler.transform(X_test.iloc[-n_rows:])
    preds_h   = best_model.predict(X_samp_sc)
    probas_h  = best_model.predict_proba(X_samp_sc)[:,1]
    hist_df   = X_test.iloc[-n_rows:].copy()
    hist_df["Predicted"] = ["🔴 HIGH" if p==1 else "🟢 LOW" for p in preds_h]
    hist_df["Prob High"] = (probas_h*100).round(1)
    hist_df["Actual"]    = y_test.iloc[-n_rows:].map({1:"🔴 HIGH",0:"🟢 LOW"}).values
    hist_df["Correct?"]  = (preds_h == y_test.iloc[-n_rows:].values)
    hist_df["Correct?"]  = hist_df["Correct?"].map({True:"✅",False:"❌"})
    st.dataframe(hist_df[["Predicted","Prob High","Actual","Correct?"]], use_container_width=True)

# ================================================================
# TAB 5 — EXPORT
# ================================================================
with tab5:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first."); st.stop()

    results   = st.session_state.results
    best_name = st.session_state.best_name
    sel_feats = st.session_state.sel_feats
    y_test    = st.session_state.y_test

    st.markdown("### 💾 Download Pipeline Artifacts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🧠 Best Model**"); st.caption(f"{best_name} (.pkl)")
        st.download_button("⬇️ Download Model",   data=st.session_state.buf_model,  file_name="malaria_model.pkl",    mime="application/octet-stream")
    with c2:
        st.markdown("**⚖️ Scaler**");             st.caption("StandardScaler (.pkl)")
        st.download_button("⬇️ Download Scaler",  data=st.session_state.buf_scaler, file_name="malaria_scaler.pkl",   mime="application/octet-stream")
    with c3:
        st.markdown("**📋 Features**");            st.caption("Feature list (.csv)")
        st.download_button("⬇️ Download Features",data=pd.Series(sel_feats,name="feature").to_csv(index=False), file_name="malaria_features.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Summary")
    summary = pd.DataFrame([{"Model":n,"Accuracy":round(i["acc"],4),"AUC-ROC":round(i["auc"],4),
                              "F1":round(i["f1"],4),"Precision":round(i["prec"],4),"Recall":round(i["rec"],4)}
                             for n, i in results.items()])
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Evaluation CSV", data=summary.to_csv(index=False),
                       file_name="model_evaluation.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 🔌 Inference Code")
    st.code("""
import joblib, pandas as pd

model    = joblib.load("malaria_model.pkl")
scaler   = joblib.load("malaria_scaler.pkl")
features = pd.read_csv("malaria_features.csv").squeeze().tolist()

def predict_risk(row: dict) -> dict:
    df    = pd.DataFrame([row])[features]
    sc    = scaler.transform(df)
    pred  = model.predict(sc)[0]
    prob  = model.predict_proba(sc)[0]
    return {
        "risk":       "HIGH RISK" if pred == 1 else "LOW RISK",
        "confidence": f"{max(prob)*100:.1f}%",
        "prob_high":  round(prob[1], 4),
    }
""", language="python")
