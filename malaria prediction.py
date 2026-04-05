import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve)


# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(page_title="Malaria Prediction | Group 3", page_icon="🦟", layout="wide")

st.markdown("""
<style>
table {color:#000000 !important; background:#ffffff !important;}
th {background:#1a5276 !important; color:#ffffff !important; padding:10px !important;}
td {color:#000000 !important; background:#ffffff !important; padding:8px !important;}
tr:nth-child(even) td {background:#eaf4fb !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1a5276,#117a65);padding:2rem;border-radius:12px;color:white;margin-bottom:1rem">
<h1>🦟 Malaria Infection Prediction System</h1>
<p>Group 3 · BSc Data Science · Meru University of Science and Technology · 2026<br>
<em>Nyanza · Rift Valley · Central Kenya</em></p>
</div>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────
FEATURES = ['Rainfall_mm','Temperature_C','Humidity_percent','Lag_1_Month_Cases',
            'Incidence_per_100k','Month','Population','Malaria_Cases',
            'Cases_Per_Capita','Region_enc','County_enc','Season_enc']
METRICS  = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
COLORS   = ['#1a5276','#117a65','#d35400']
MN       = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MODELS_AVAILABLE = ["Logistic Regression","Random Forest","Gradient Boosting"]
SAVE_DIR = "/tmp/malaria_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── SIDEBAR: FILE UPLOAD ──────────────────────────────────────────
st.sidebar.header("📂 Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# ── DATA LOADING ──────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if not uploaded:
    st.info("⬅️ Upload Final_Malaria_Dataset.csv in the sidebar to begin.")
    st.stop()

df_raw = load_data(uploaded)


# ── PREPROCESSING ─────────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()

    for c in ['ID','Health_Facilities','Avg_Income','Disease_Cases','Notes']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    df.drop_duplicates(inplace=True)

    df['Region'] = df['Region'].str.strip().str.title()
    df['County'] = df['County'].str.strip().str.title()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

    for col in ['Rainfall_mm','Temperature_C','Humidity_percent',
                'Malaria_Cases','Lag_1_Month_Cases','Incidence_per_100k']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR    = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    def season(m):
        return ('Long_Rains'  if m in [3,4,5]   else
                'Dry'         if m in [6,7,8]   else
                'Short_Rains' if m in [9,10,11] else 'Cool_Dry')
    df['Season'] = df['Month'].apply(season)

    df['Cases_Per_Capita'] = df['Malaria_Cases'] / df['Population'] * 100000

    df['Region_enc'] = LabelEncoder().fit_transform(df['Region'])
    df['County_enc'] = LabelEncoder().fit_transform(df['County'])
    df['Season_enc'] = LabelEncoder().fit_transform(df['Season'])

    return df[FEATURES], df['High_Risk_Binary'].astype(int)

X, y = preprocess(df_raw)


# ── DATASET OVERVIEW ──────────────────────────────────────────────
st.subheader("📊 Dataset Overview")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Records",        f"{len(X):,}")
c2.metric("Features",       len(FEATURES))
c3.metric("Low Risk",       f"{int((y==0).sum()):,}")
c4.metric("High Risk",      f"{int((y==1).sum()):,}")
c5.metric("High-Risk Rate", f"{y.mean()*100:.1f}%")


# ── SIDEBAR: MODEL SETTINGS ───────────────────────────────────────
st.sidebar.header("⚙️ Settings")
test_size  = st.sidebar.slider("Test split %", 10, 40, 20) / 100
models_sel = st.sidebar.multiselect("Models", MODELS_AVAILABLE, default=MODELS_AVAILABLE)

if not models_sel:
    st.warning("Select at least one model."); st.stop()


# ── MODEL TRAINING ────────────────────────────────────────────────
models_trained = all(
    os.path.exists(f"{SAVE_DIR}/{n.replace(' ','_')}.pkl") for n in models_sel
)

if st.sidebar.button("🚀 Train Models", type="primary") or not models_trained:
    st.subheader("🔧 Training Models...")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    sc      = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    joblib.dump(sc, f"{SAVE_DIR}/scaler.pkl")

    perf = {}
    prog = st.progress(0)

    for i, name in enumerate(models_sel):
        st.write(f"⏳ Training {name}...")

        if name == "Logistic Regression":
            m = LogisticRegression(max_iter=1000, random_state=42)
            m.fit(X_tr_sc, y_tr)
            yp  = m.predict(X_te_sc)
            ypr = m.predict_proba(X_te_sc)[:,1]

        elif name == "Random Forest":
            m = RandomForestClassifier(n_estimators=200, random_state=42)
            m.fit(X_tr, y_tr)
            yp  = m.predict(X_te)
            ypr = m.predict_proba(X_te)[:,1]

        else:
            m = GradientBoostingClassifier(n_estimators=200, random_state=42)
            m.fit(X_tr, y_tr)
            yp  = m.predict(X_te)
            ypr = m.predict_proba(X_te)[:,1]

        perf[name] = {
            'Accuracy':  round(accuracy_score(y_te, yp),  4),
            'Precision': round(precision_score(y_te, yp, zero_division=0), 4),
            'Recall':    round(recall_score(y_te,    yp, zero_division=0), 4),
            'F1 Score':  round(f1_score(y_te,        yp, zero_division=0), 4),
            'ROC-AUC':   round(roc_auc_score(y_te,  ypr), 4),
            'yp':  yp.tolist(),
            'ypr': ypr.tolist(),
            'cm':  confusion_matrix(y_te, yp).tolist(),
            'fi':  m.feature_importances_.tolist()
                   if hasattr(m, 'feature_importances_') else None
        }

        joblib.dump(m, f"{SAVE_DIR}/{name.replace(' ','_')}.pkl")
        prog.progress((i+1) / len(models_sel))

    joblib.dump(perf,           f"{SAVE_DIR}/perf.pkl")
    joblib.dump(y_te.tolist(),  f"{SAVE_DIR}/y_te.pkl")
    st.success("✅ All models trained and saved!")


# ── LOAD SAVED RESULTS ────────────────────────────────────────────
if not os.path.exists(f"{SAVE_DIR}/perf.pkl"):
    st.info("👈 Click **Train Models** in the sidebar to begin.")
    st.stop()

perf      = joblib.load(f"{SAVE_DIR}/perf.pkl")
y_te_list = joblib.load(f"{SAVE_DIR}/y_te.pkl")
y_te_arr  = np.array(y_te_list)

trained = [n for n in models_sel if n in perf]
if not trained:
    st.info("👈 Click **Train Models** to train the selected models.")
    st.stop()


# ── PERFORMANCE SUMMARY TABLE ─────────────────────────────────────
st.subheader("📈 Performance Summary")
best = max(trained, key=lambda n: perf[n]['F1 Score'])

rows = ""
for idx, name in enumerate(trained):
    ib  = (name == best)
    bg  = '#d5f5e3' if ib else ('#eaf4fb' if idx%2==0 else '#ffffff')
    fw  = 'bold' if ib else 'normal'
    tag = '  ⭐ Best' if ib else ''
    cells = ''.join(
        f'<td style="padding:10px;border:1px solid #999;color:#000;text-align:center;">'
        f'{perf[name][m]:.4f}</td>' for m in METRICS
    )
    rows += (f'<tr style="background:{bg};">'
             f'<td style="padding:10px;border:1px solid #999;color:#000;font-weight:{fw};">'
             f'{name}{tag}</td>{cells}</tr>')

hdrs = ''.join(
    f'<th style="padding:10px;border:1px solid #999;text-align:center;">{m}</th>'
    for m in METRICS
)
st.markdown(
    f'<table style="width:100%;border-collapse:collapse;font-size:15px;">'
    f'<tr style="background:#1a5276;color:#fff;">'
    f'<th style="padding:10px;border:1px solid #999;text-align:left;">Model</th>'
    f'{hdrs}</tr>{rows}</table>',
    unsafe_allow_html=True
)
st.success(f"🏆 Best Model: **{best}** — F1 = {perf[best]['F1 Score']:.4f}")


# ── CHARTS ────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📊 Metrics","🟦 Confusion Matrices","📉 ROC Curves"])

with t1:
    fig, ax = plt.subplots(figsize=(11,5))
    x = np.arange(len(METRICS))
    w = 0.7 / len(trained)
    for i, name in enumerate(trained):
        vals = [perf[name][m] for m in METRICS]
        bars = ax.bar(
            x + i*w - (len(trained)-1)*w/2, vals, w,
            label=name, color=COLORS[i%3], alpha=0.88, edgecolor='white'
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(METRICS); ax.set_ylim(0,1.15)
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with t2:
    cols = st.columns(len(trained))
    for i, name in enumerate(trained):
        cm = np.array(perf[name]['cm'])
        fig, ax = plt.subplots(figsize=(4,3.5))
        im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Low','High'], rotation=30)
        ax.set_yticklabels(['Low','High'])
        thresh = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[r,c] > thresh else 'black')
        ax.set_title(name, fontweight='bold')
        plt.tight_layout(); cols[i].pyplot(fig); plt.close()

with t3:
    fig, ax = plt.subplots(figsize=(8,6))
    for i, name in enumerate(trained):
        fpr, tpr, _ = roc_curve(y_te_arr, np.array(perf[name]['ypr']))
        ax.plot(fpr, tpr, lw=2.5, color=COLORS[i%3],
                label=f'{name} (AUC={perf[name]["ROC-AUC"]:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random baseline')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold'); ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ── FEATURE IMPORTANCE ────────────────────────────────────────────
if "Random Forest" in trained and perf["Random Forest"]['fi']:
    st.subheader("🌲 Feature Importance (Random Forest)")
    fi = pd.Series(perf['Random Forest']['fi'], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(fi.index, fi.values,
            color=['#c0392b' if v >= fi.quantile(0.75) else '#1a5276'
                   for v in fi.values],
            alpha=0.88, edgecolor='white')
    ax.set_title('Feature Importance — top features shown in red', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ── LIVE PREDICTION PANEL ─────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Live Prediction Panel")
st.markdown("🎛️ **Adjust any slider below — predictions update automatically in real time.**")

def enc_season(m):
    s = ('Long_Rains'  if m in [3,4,5]   else
         'Dry'         if m in [6,7,8]   else
         'Short_Rains' if m in [9,10,11] else 'Cool_Dry')
    return {'Cool_Dry':0,'Dry':1,'Long_Rains':2,'Short_Rains':3}[s], s

st.markdown("#### 🌦️ Climate Conditions")
p1, p2, p3 = st.columns(3)
rainfall    = p1.slider("🌧️ Rainfall (mm)",     0.0, 350.0, 150.0, 5.0,  key="rf")
temperature = p2.slider("🌡️ Temperature (°C)", 10.0,  40.0,  24.0, 0.5,  key="tmp")
humidity    = p3.slider("💧 Humidity (%)",       0.0, 100.0,  70.0, 1.0,  key="hum")

st.markdown("#### 🦟 Epidemiological Inputs")
p4, p5, p6 = st.columns(3)
lag_cases   = p4.slider("📅 Lag 1 Month Cases",  0.0, 3500.0, 1200.0, 50.0, key="lag")
incidence   = p5.slider("📊 Incidence per 100k", 0.0,  600.0,  120.0,  5.0, key="inc")
month       = p6.selectbox("📆 Month", list(range(1,13)),
                            format_func=lambda m:MN[m-1], index=5, key="mon")

st.markdown("#### 👥 Population")
p7, p8 = st.columns(2)
population  = p7.slider("🏘️ Population",    100000, 3000000, 500000, 10000, key="pop")
mal_cases   = p8.slider("🦟 Malaria Cases",    0.0,  3500.0, 1200.0,  50.0, key="mc")


# ── LIVE PREDICTION COMPUTATION ───────────────────────────────────
se, sn = enc_season(month)
cpc    = mal_cases / max(population, 1) * 100000

inp = pd.DataFrame(
    [[rainfall, temperature, humidity, lag_cases, incidence,
      month, population, mal_cases, cpc, 0, 0, se]],
    columns=FEATURES
)

sc_loaded = joblib.load(f"{SAVE_DIR}/scaler.pkl")

st.markdown("---")
st.markdown("### 🎯 Live Prediction Results")
st.caption(f"📍 Season: **{sn.replace('_',' ')}**  |  Cases per capita: **{cpc:.1f}** per 100k")

probs = []
rcols = st.columns(len(trained))
for i, name in enumerate(trained):
    m    = joblib.load(f"{SAVE_DIR}/{name.replace(' ','_')}.pkl")
    X_in = sc_loaded.transform(inp) if name == "Logistic Regression" else inp
    pred = m.predict(X_in)[0]
    prob = m.predict_proba(X_in)[0][1]
    probs.append(prob)
    bg  = '#c0392b' if pred == 1 else '#117a65'
    lbl = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
    rcols[i].markdown(f"""
    <div style="background:{bg};padding:1.4rem;border-radius:12px;
                text-align:center;color:white;margin:4px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);">
        <div style="font-size:0.85rem;opacity:0.85;margin-bottom:4px;">{name}</div>
        <div style="font-size:1.6rem;font-weight:bold;margin:0.3rem 0;">{lbl}</div>
        <div style="font-size:1.1rem;">Confidence: <b>{prob*100:.1f}%</b></div>
    </div>""", unsafe_allow_html=True)

avg = np.mean(probs)

fig, ax = plt.subplots(figsize=(8,4))
bar_colors = ['#c0392b' if p >= 0.5 else '#117a65' for p in probs]
bars = ax.bar(trained, [p*100 for p in probs],
              color=bar_colors, edgecolor='white', alpha=0.9, width=0.45)
ax.axhline(50, color='grey', lw=1.5, linestyle='--', alpha=0.7,
           label='50% Risk Threshold')
for bar, p in zip(bars, probs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{p*100:.1f}%', ha='center', va='bottom',
            fontsize=13, fontweight='bold',
            color='#c0392b' if p >= 0.5 else '#117a65')
ax.set_ylim(0,120)
ax.set_ylabel('High-Risk Probability (%)', fontsize=11)
ax.set_title(f'Live Risk Probability  |  Ensemble Average: {avg*100:.1f}%',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); st.pyplot(fig); plt.close()

risk_label = "🔴 HIGH RISK" if avg >= 0.5 else "🟢 LOW RISK"
gauge_col  = '#c0392b'      if avg >= 0.5 else '#117a65'
fig, ax = plt.subplots(figsize=(10,1.5))
ax.barh([0],[1], color='#ecf0f1', height=0.6)
ax.barh([0],[avg], color=gauge_col, height=0.6, alpha=0.9)
ax.axvline(0.5, color='#555', lw=2.5, linestyle='--')
ax.set_xlim(0,1); ax.set_yticks([])
ax.set_xticks([0,0.25,0.5,0.75,1.0])
ax.set_xticklabels(['0%','25%','50% threshold','75%','100%'], fontsize=10)
ax.set_title(f'Ensemble Risk Score: {avg*100:.1f}%   →   {risk_label}',
             fontweight='bold', fontsize=13, color=gauge_col)
ax.spines[['top','right','left']].set_visible(False)
plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("#### 📋 Input Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("🌧️ Rainfall",    f"{rainfall:.0f} mm",   delta="High ⚠️"   if rainfall    > 200  else "Normal ✅")
s2.metric("🌡️ Temperature", f"{temperature:.1f}°C", delta="High ⚠️"   if temperature > 30   else "Normal ✅")
s3.metric("💧 Humidity",    f"{humidity:.0f}%",      delta="High ⚠️"   if humidity    > 80   else "Normal ✅")
s4.metric("🦟 Lag Cases",   f"{lag_cases:.0f}",      delta="High ⚠️"   if lag_cases   > 1500 else "Normal ✅")


# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Group 3 | BSc Data Science | Meru University of Science and Technology | 2026")
