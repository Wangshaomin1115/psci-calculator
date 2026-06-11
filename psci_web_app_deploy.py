"""
PSCI Risk Prediction Web App — Deployment Version
===================================================
Deploy on Streamlit Cloud:
  1. Upload this file + cox_final.pkl + preprocessor.pkl + feature_cols.pkl to GitHub
  2. Connect repo to share.streamlit.io
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 路径配置（相对路径，适合云端部署）
# ============================================================

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "cox_final.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_cols.pkl")

FEATURE_DEFAULTS = {
    "age": 67.0, "gender": 1.0, "education": 2.0,
    "marital_status": 1.0, "hypertension": 0.0, "diabetes": 0.0,
    "heart_disease": 0.0, "bmi": 24.5, "smoke_ever": 0.0,
    "smoke_current": 0.0, "physical_activity": 2.0, "alcohol_use": 0.0,
    "depression_cesd": 5.0, "cog_z_unified": 0.0, "memory_z": 0.0,
    "orient_z": 0.0, "adl_score": 0.0, "iadl_score": 0.0,
    "self_rated_health": 3.0, "grip_strength": 28.0,
    "hearing_difficulty": 0.0, "urban_rural": 1.0,
}

PRED_TIMES = [1, 3, 5]

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="PSCI Risk Calculator",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 98% !important; padding-left: 2rem; padding-right: 2rem; }
    .metric-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px 20px; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #0f3460; }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .risk-high   { color: #c0392b; font-weight: 700; }
    .risk-medium { color: #e67e22; font-weight: 700; }
    .risk-low    { color: #27ae60; font-weight: 700; }
    .note-box { background: #f0f4ff; border-left: 4px solid #4a6cf7; padding: 10px 16px; border-radius: 4px; font-size: 0.85rem; color: #444; margin-top: 8px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 加载模型
# ============================================================

@st.cache_resource
def load_model():
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, preprocessor, feature_cols

try:
    cox_model, preprocessor, feature_cols = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ============================================================
# 特征配置
# ============================================================

FEATURES = [
    {"key": "age",             "label": "Age at stroke (years)",                    "sig": "***", "min": 40,   "max": 95,  "default": 65,  "step": 1,   "help": "Patient's age at the time of stroke"},
    {"key": "cog_z_unified",   "label": "Pre-stroke cognition – composite Z-score", "sig": "***", "min": -4.0, "max": 4.0, "default": 0.0, "step": 0.1, "help": "Standardised composite cognitive Z-score before stroke (mean=0, SD=1)"},
    {"key": "physical_activity","label": "Physical activity score",                  "sig": "***", "min": 0,    "max": 5,   "default": 2,   "step": 1,   "help": "Physical activity level (0–5 scale; higher = more active)"},
    {"key": "depression_cesd", "label": "Depression – CESD score",                  "sig": "**",  "min": 0,    "max": 30,  "default": 5,   "step": 1,   "help": "Center for Epidemiologic Studies Depression scale (higher = more depressed)"},
    {"key": "self_rated_health","label": "Self-rated health (1–5)",                  "sig": "*",   "min": 1,    "max": 5,   "default": 3,   "step": 1,   "help": "Self-rated health: 1=Excellent, 5=Poor"},
    {"key": "grip_strength",   "label": "Grip strength (kg)",                       "sig": "*",   "min": 5.0,  "max": 70.0,"default": 28.0,"step": 0.5, "help": "Maximum grip strength measured by dynamometer (kg)"},
]

SIG_COLOR = {"***": "#c0392b", "**": "#e67e22", "*": "#f39c12"}

# ============================================================
# 页面标题
# ============================================================

st.markdown("## 🧠 Post-Stroke Cognitive Impairment (PSCI) Risk Calculator")
st.markdown(
    "This tool estimates individualised PSCI risk using a validated Cox proportional-hazards model "
    "trained on six international longitudinal cohorts (CHARLS, HRS, ELSA, MHAS, SHARE, KLOSA; N = 3,381). "
    "Enter patient characteristics below to generate a risk prediction."
)

if not model_loaded:
    st.error(f"⚠️ Model file not found. Error: {load_error}")
    st.stop()

st.divider()

# ============================================================
# 三栏布局
# ============================================================

col_input, col_result, col_shap = st.columns([1, 1.1, 1], gap="large")

# ── 左栏：输入 ───────────────────────────────────────────────
with col_input:
    st.markdown("### Patient Characteristics")
    st.markdown(
        '<div class="note-box">Significance in multivariable Cox model: '
        '<span style="color:#c0392b">*** p&lt;0.001</span> &nbsp;'
        '<span style="color:#e67e22">** p&lt;0.01</span> &nbsp;'
        '<span style="color:#f39c12">* p&lt;0.05</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    input_vals = {}
    for feat in FEATURES:
        st.markdown(
            f"**{feat['label']}** "
            f'<span style="color:{SIG_COLOR[feat["sig"]]};font-weight:700">{feat["sig"]}</span>',
            unsafe_allow_html=True
        )
        val = st.slider(
            label=feat["label"], min_value=float(feat["min"]),
            max_value=float(feat["max"]), value=float(feat["default"]),
            step=float(feat["step"]), help=feat["help"],
            label_visibility="collapsed",
        )
        input_vals[feat["key"]] = val

    predict_btn = st.button("🔍 Calculate Risk", type="primary", use_container_width=True)

# ============================================================
# 预测函数
# ============================================================

def build_input(input_vals):
    row = dict(FEATURE_DEFAULTS)
    for feat in FEATURES:
        row[feat["key"]] = input_vals[feat["key"]]
    df_raw = pd.DataFrame([row])[feature_cols]
    return preprocessor.transform(df_raw)

def get_risk_label(p):
    if p >= 0.4:   return "High Risk",     "risk-high"
    if p >= 0.2:   return "Moderate Risk", "risk-medium"
    return "Low Risk", "risk-low"

def plot_curve(sf_pred):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    t = sf_pred.index.values
    r = (1 - sf_pred.values.flatten()) * 100
    ax.plot(t, r, color="#e74c3c", lw=2.5)
    ax.fill_between(t, 0, r, alpha=0.12, color="#e74c3c")
    for pt in PRED_TIMES:
        idx = min(np.searchsorted(t, pt), len(t)-1)
        ax.axvline(pt, color="#bbb", ls="--", lw=1)
        ax.scatter([t[idx]], [r[idx]], color="#c0392b", s=55, zorder=4)
        ax.annotate(f"{r[idx]:.1f}%", xy=(t[idx], r[idx]),
                    xytext=(t[idx]+0.15, r[idx]+2),
                    fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel("Years since stroke", fontsize=10)
    ax.set_ylabel("Cumulative PSCI risk (%)", fontsize=10)
    ax.set_title("Predicted Cumulative PSCI Risk", fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(0, min(100, ax.get_ylim()[1]*1.15))
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def plot_waterfall(coef_contrib):
    items = sorted(
        [{"label": f["label"], "val": coef_contrib.get(f["key"], 0.0)} for f in FEATURES],
        key=lambda x: x["val"]
    )
    labels  = [i["label"] for i in items]
    vals    = [i["val"]   for i in items]
    colors  = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig, ax = plt.subplots(figsize=(6, max(5.0, len(labels)*0.75)))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor="white")
    x_max = max(abs(v) for v in vals) if vals else 0.1
    for bar, val in zip(bars, vals):
        label  = f"{val:+.3f}"
        bar_len = abs(bar.get_width())
        char_w  = x_max * 0.018 * len(label)
        if bar_len >= char_w * 1.2:
            ax.text(bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                    label, va="center", ha="center",
                    fontsize=8.5, color="white", fontweight="bold")
        else:
            offset = x_max*0.03 if val >= 0 else -x_max*0.03
            ax.text(bar.get_width()+offset, bar.get_y()+bar.get_height()/2,
                    label, va="center", ha="left" if val >= 0 else "right",
                    fontsize=8.5, color="#333", fontweight="bold")
    ax.axvline(0, color="black", lw=1, alpha=0.6)
    ax.set_xlabel("Contribution to log Hazard Ratio", fontsize=9)
    ax.set_title("Feature Contributions (Waterfall)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

# ── 中栏：预测结果 ───────────────────────────────────────────
with col_result:
    st.markdown("### Prediction Results")
    if not predict_btn:
        st.info("👈 Adjust patient characteristics on the left, then click **Calculate Risk**.")
    else:
        with st.spinner("Computing..."):
            try:
                X = build_input(input_vals)
                sf_fn  = cox_model.predict_survival_function(X)[0]
                sf_pred = pd.Series(sf_fn.y, index=sf_fn.x)

                risk_at = {}
                for t in PRED_TIMES:
                    idx = min(np.searchsorted(sf_pred.index.values, t), len(sf_pred)-1)
                    risk_at[t] = float(1 - sf_pred.iloc[idx])

                risk_label, risk_css = get_risk_label(risk_at[3])

                cols_m = st.columns(len(PRED_TIMES))
                for i, t in enumerate(PRED_TIMES):
                    with cols_m[i]:
                        st.markdown(
                            f'<div class="metric-box">'
                            f'<div class="metric-value">{risk_at[t]*100:.1f}%</div>'
                            f'<div class="metric-label">{t}-year PSCI risk</div>'
                            f'</div>', unsafe_allow_html=True)

                st.markdown("")
                st.markdown(
                    f'**Risk category (3-year):** <span class="{risk_css}">{risk_label}</span>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="note-box">Low: &lt;20% &nbsp;|&nbsp; Moderate: 20–40% &nbsp;|&nbsp; High: ≥40%</div>',
                    unsafe_allow_html=True)
                st.markdown("")

                fig_surv = plot_curve(sf_pred)
                st.pyplot(fig_surv, use_container_width=True)
                plt.close()

                # 计算特征贡献
                try:
                    proc_names = preprocessor.get_feature_names_out()
                    proc_coefs = cox_model.coef_
                    X_df = pd.DataFrame(X, columns=proc_names)
                    coef_contrib = {}
                    for feat in FEATURES:
                        key = feat["key"]
                        matched = [i for i, n in enumerate(proc_names)
                                   if (n.split("__",1)[1] if "__" in n else n) == key
                                   or (n.split("__",1)[1] if "__" in n else n).startswith(key+"_")]
                        coef_contrib[key] = sum(proc_coefs[i]*float(X_df.iloc[0,i]) for i in matched)
                except Exception:
                    coef_contrib = {f["key"]: 0.0 for f in FEATURES}

                fig_wf = plot_waterfall(coef_contrib)
                st.session_state["fig_wf"] = fig_wf

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ── 右栏：Waterfall ──────────────────────────────────────────
with col_shap:
    st.markdown("### Feature Contributions")
    if predict_btn and "fig_wf" in st.session_state:
        st.pyplot(st.session_state["fig_wf"], use_container_width=True)
        plt.close()
        st.markdown(
            '<div class="note-box">'
            '<b>Red bars</b>: ↑ PSCI risk &nbsp;|&nbsp; <b>Blue bars</b>: ↓ PSCI risk<br>'
            'Each bar = model coefficient × standardised feature value'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown("")
        st.info("Feature contributions will appear here after clicking **Calculate Risk**.")

# ============================================================
# 底部说明
# ============================================================

st.divider()
st.markdown("""
<div style="font-size:0.8rem; color:#888; line-height:1.6">
<b>Model information:</b>
Cox PH model trained on 2,704 participants (CHARLS, HRS, ELSA, MHAS; 80% split).
Internal C-index = 0.776 (95% CI 0.736–0.816); External C-index = 0.742 (95% CI 0.709–0.776; SHARE n=2,077).
<br><br>
<b>Disclaimer:</b>
This tool is intended for research purposes only and should not replace clinical judgement.
Predictions are based on population-level models and may not apply to individual patients.
</div>
""", unsafe_allow_html=True)
