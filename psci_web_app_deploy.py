# -*- coding: utf-8 -*-
"""
PSCI Risk Prediction Web App — v2 (17-predictor model)
======================================================
配套 step_D_main_analysis.py (v4_FINAL) 的部署模型。

与旧版的区别（重要）：
  * 模型文件改为 cox_deployment.pkl / preprocessor_deployment.pkl
  * 预测因子从 22 个改为 17 个；physical_activity、grip_strength、memory_z、
    orient_z、urban_rural 已被排除，旧版界面里的滑块有一半不在模型里了
  * depression 从 CESD 连续分改为二分类 depressive_symptoms
  * education 改为两个哑变量（level2 / level3，参照 = level 1）
  * 17 个预测因子全部开放输入，默认值来自开发队列的中位数/众数
  * 显著性、C-index、队列规模全部从 deployment_reference.json 读取，不硬编码

本地运行：
    pip install -r requirements.txt
    streamlit run psci_web_app.py

云端部署：
    把 deploy/ 整个目录（含 3 个 pkl/json + 本文件 + requirements.txt）传 GitHub，
    连到 share.streamlit.io。
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BUNDLE_FILES = ("cox_deployment.pkl", "preprocessor_deployment.pkl",
                "deployment_reference.json")
DEATH_MODEL_FILE = "cox_death_deployment.pkl"   # 可选：有它才能输出 CIF


def find_bundle_dir():
    """按优先级寻找部署包所在目录。

    支持三种放法：
      1) 三个文件和本脚本放在一起（云端部署时的标准做法）
      2) 本脚本在 PSCI\\scripts\\ 下，文件在 PSCI\\deploy\\ 下
      3) 用环境变量 PSCI_DEPLOY_DIR 显式指定
    """
    candidates = []
    env = os.environ.get("PSCI_DEPLOY_DIR")
    if env:
        candidates.append(os.path.abspath(env))
    candidates += [
        BASE_DIR,                                                  # 同目录
        os.path.join(os.path.dirname(BASE_DIR), "deploy"),         # ../deploy
        os.path.join(BASE_DIR, "deploy"),                          # ./deploy
        os.path.join(os.getcwd(), "deploy"),                       # 当前工作目录/deploy
        os.getcwd(),
    ]
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        if all(os.path.exists(os.path.join(d, f)) for f in BUNDLE_FILES):
            return d, candidates
    return None, candidates


# 各队列为两年一访，中风后第 1 年几乎没有认知复测，
# 报告 1 年风险等于报告"还没测到"，因此从 3 年起。
PRED_TIMES = [3, 5, 10]

# =============================================================================
# 17 个预测因子的界面定义（key 必须与 design_order 完全一致）
# =============================================================================

DEMOGRAPHIC = [
    # key 用对外名称 "age"；真实模型列名 age_at_stroke_h 由 deployment_reference.json
    # 里的 public_aliases 负责映射，只在 build_X() 里调 transform 前一刻还原。
    {"key": "age", "label": "Age at stroke (years)", "kind": "slider",
     "step": 1.0, "help": "Age at the first stroke-positive survey wave"},
    {"key": "gender", "label": "Sex", "kind": "binary",
     "yes": "Male", "no": "Female", "help": "Coded 1 = male in the model"},
    {"key": "married_partnered", "label": "Married or partnered", "kind": "binary",
     "yes": "Yes", "no": "No", "help": ""},
]

COGNITION = [
    {"key": "cog_z_unified", "label": "Pre-stroke cognition (composite Z)", "kind": "slider",
     "step": 0.1, "help": "Standardised pre-stroke cognitive composite (cohort mean 0, SD 1)"},
]

COMORBID = [
    {"key": "hypertension", "label": "Hypertension", "kind": "binary",
     "yes": "Yes", "no": "No",
     "help": "Self-reported physician-diagnosed hypertension before stroke"},
    {"key": "diabetes", "label": "Diabetes", "kind": "binary", "yes": "Yes", "no": "No", "help": ""},
    {"key": "heart_disease_h", "label": "Heart disease / heart attack", "kind": "binary",
     "yes": "Yes", "no": "No", "help": ""},
    {"key": "bmi_model", "label": "BMI (kg/m²)", "kind": "slider",
     "step": 0.1, "help": "Most recent valid measurement before the stroke"},
]

LIFESTYLE = [
    {"key": "smoke_ever", "label": "Ever smoked", "kind": "binary", "yes": "Yes", "no": "No", "help": ""},
    {"key": "smoke_current", "label": "Current smoker", "kind": "binary", "yes": "Yes", "no": "No", "help": ""},
    {"key": "alcohol_use", "label": "Alcohol use", "kind": "binary", "yes": "Yes", "no": "No", "help": ""},
]

FUNCTION = [
    {"key": "adl_common5", "label": "ADL difficulties (0–5)", "kind": "slider",
     "step": 1.0, "help": "Number of basic activities of daily living with difficulty"},
    {"key": "iadl_common4", "label": "IADL difficulties (0–4)", "kind": "slider",
     "step": 1.0, "help": "Number of instrumental activities of daily living with difficulty"},
    {"key": "self_rated_health", "label": "Self-rated health (1 excellent – 5 poor)", "kind": "slider",
     "step": 1.0, "help": "1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor"},
    {"key": "hearing_h", "label": "Hearing (1 excellent – 5 poor)", "kind": "slider",
     "step": 1.0, "help": "1 = Excellent ... 5 = Poor"},
    {"key": "depressive_symptoms", "label": "Elevated depressive symptoms", "kind": "binary",
     "yes": "Yes", "no": "No", "help": "Above the cohort-specific depression screening cut-off"},
]

GROUPS = [
    ("Demographics", DEMOGRAPHIC),
    ("Cognition", COGNITION),
    ("Comorbidity", COMORBID),
    ("Lifestyle", LIFESTYLE),
    ("Function & mood", FUNCTION),
]

# education 单独处理：3 级下拉 -> education_level2 / education_level3
EDU_OPTIONS = {
    "Level 1 — less than upper secondary (reference)": (0.0, 0.0),
    "Level 2 — upper secondary / vocational": (1.0, 0.0),
    "Level 3 — tertiary": (0.0, 1.0),
}

# =============================================================================
# 加载
# =============================================================================

st.set_page_config(page_title="PSCI Risk Calculator", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 98% !important;
                   padding-left: 2rem; padding-right: 2rem; }
.metric-box { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px;
              padding:14px 16px; text-align:center; }
.metric-value { font-size:1.8rem; font-weight:700; color:#0f3460; }
.metric-label { font-size:0.8rem; color:#6c757d; margin-top:4px; }
.risk-high{color:#c0392b;font-weight:700}
.risk-medium{color:#e67e22;font-weight:700}
.risk-low{color:#27ae60;font-weight:700}
.note-box { background:#f0f4ff; border-left:4px solid #4a6cf7; padding:10px 16px;
            border-radius:4px; font-size:0.85rem; color:#444; margin-top:8px; }
.warn-box { background:#fff8e6; border-left:4px solid #e6a700; padding:10px 16px;
            border-radius:4px; font-size:0.85rem; color:#5a4500; margin-top:8px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle(bundle_dir):
    model = joblib.load(os.path.join(bundle_dir, "cox_deployment.pkl"))
    prep = joblib.load(os.path.join(bundle_dir, "preprocessor_deployment.pkl"))
    with open(os.path.join(bundle_dir, "deployment_reference.json"), "r", encoding="utf-8") as f:
        ref = json.load(f)
    dpath = os.path.join(bundle_dir, DEATH_MODEL_FILE)
    death = joblib.load(dpath) if os.path.exists(dpath) else None
    return model, prep, ref, death


BUNDLE_DIR, TRIED = find_bundle_dir()

st.markdown("## 🧠 Post-Stroke Cognitive Impairment (PSCI) Risk Calculator")

if BUNDLE_DIR is None:
    msg = (
        "找不到部署文件。需要以下三个文件放在同一个目录：\n"
        + "".join(f"  - {f}\n" for f in BUNDLE_FILES)
        + "\n已尝试的目录：\n"
        + "".join(f"  - {d}\n" for d in TRIED)
        + "\n解决办法：\n"
        "  1. 先运行  python export_deployment_bundle.py  生成 PSCI\\deploy\\ 目录；\n"
        "  2. 把 psci_web_app.py 复制进 PSCI\\deploy\\，然后在该目录下运行\n"
        "         streamlit run psci_web_app.py\n"
        "     或者用环境变量指定：  set PSCI_DEPLOY_DIR=D:\\生信\\csv版本清洗后\\PSCI\\deploy\n"
    )
    print("\n[ERROR] " + msg)          # bare mode 下也能看到
    st.error(msg)
    st.stop()
    raise SystemExit(1)               # python 直接运行时 st.stop() 不生效，这行兜底

try:
    cox_model, preprocessor, REF, death_model = load_bundle(BUNDLE_DIR)
except Exception as e:
    msg = (
        f"部署文件加载失败（目录 {BUNDLE_DIR}）：{e}\n\n"
        "最常见原因是 scikit-learn / scikit-survival 版本与训练时不一致，"
        "导致 pkl 反序列化失败。请把训练环境的版本号写进 requirements.txt。"
    )
    print("\n[ERROR] " + msg)
    st.error(msg)
    st.stop()
    raise SystemExit(1)

DESIGN_ORDER = REF["design_order"]          # 真实模型列名，顺序即 pkl 期望的顺序
CONTINUOUS = set(REF["continuous"])
DEFAULTS = REF["reference_values"]           # 按真实列名索引
RANGES = REF["input_ranges"]                 # 按真实列名索引
COEFS = REF.get("coefficients", {})          # 按真实列名索引
PERF = REF.get("performance", {})
COH = REF.get("cohort", {})

# ---- 对外名称 <-> 模型列名 ------------------------------------------------
# preprocessor 是按列名选列的，输入 DataFrame 的列名必须和训练时一模一样，
# 所以这里只做一层别名，绝不改动送进模型的真实列名。
CENTERS = REF.get("centering_values", {})   # 变换后空间的队列列均值
P_FLOOR = REF.get("p_value_floor")          # bootstrap P 值的理论下限
MERGE_BARS = REF.get("merge_bars", {})      # waterfall 合并显示的哑变量组
RISK_CUTS = REF.get("risk_cutpoints")       # 开发队列预测 5 年 CIF 的三分位
CIF_MODE = bool(REF.get("cif_mode")) and death_model is not None

MODEL_TO_PUBLIC = REF.get("public_aliases", {})
PUBLIC_TO_MODEL = {v: k for k, v in MODEL_TO_PUBLIC.items()}


def mkey(k):
    """把界面上的对外名称解析成模型真实列名。"""
    return PUBLIC_TO_MODEL.get(k, k)


def pkey(k):
    """把模型真实列名转成对外名称。"""
    return MODEL_TO_PUBLIC.get(k, k)

dev_names = ", ".join(COH.get("development_cohorts", []))
ext_names = ", ".join(COH.get("external_cohort", []))
st.markdown(
    f"Individualised PSCI risk from a Cox proportional-hazards model with 17 predictors, "
    f"**all measured before the stroke**, developed in {dev_names} "
    f"(n = {COH.get('n_development','?'):,}) and externally validated in {ext_names} "
    f"(n = {COH.get('n_external','?'):,}). PSCI is defined as a post-stroke cognitive "
    f"Z score below −2.0 within {COH.get('max_follow_up_years', 10)} years."
    + (" Estimates are cumulative incidences with death before PSCI treated as a "
       "competing event." if CIF_MODE else "")
)

st.divider()


def stretch(fn, *args, **kwargs):
    """Streamlit 1.49+ 用 width='stretch'，旧版用 use_container_width=True。
    两种都试，避免在不同 streamlit 版本上报 TypeError 或刷弃用警告。"""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def sig_stars(key):
    c = COEFS.get(mkey(key))
    if not c:
        return "", "#999"
    p = c["p"]
    # bootstrap 符号检验的 P 有理论下限 2/(B+1)（B=1000 时为 0.002），
    # 落在下限上的 P 实际是"小于该值"，不能按 0.002 归入 ** 档。
    if P_FLOOR and p <= P_FLOOR * 1.02:   # 容忍 CSV 里的四舍五入
        return "***", "#c0392b"
    if p < 0.001:
        return "***", "#c0392b"
    if p < 0.01:
        return "**", "#e67e22"
    if p < 0.05:
        return "*", "#f39c12"
    return "", "#999"


def hr_text(key):
    c = COEFS.get(mkey(key))
    if not c:
        return ""
    return f"HR {c['HR']:.2f} ({c['HR_ci_lo']:.2f}–{c['HR_ci_hi']:.2f}), P = {c['p']:.3f}"


# =============================================================================
# 输入
# =============================================================================

col_in, col_out, col_wf = st.columns([1, 1.15, 1], gap="large")

with col_in:
    st.markdown("### Patient characteristics")
    star_note = ""
    if P_FLOOR:
        b = int(round(2.0 / P_FLOOR)) - 1
        star_note = (f'P values come from a {b}-replicate bootstrap sign test whose lower '
                     f'limit is {P_FLOOR}, so *** means "at or below that limit". ')
    st.markdown(
        '<div class="note-box">Stars show significance in the primary multivariable Cox model: '
        f'<span style="color:#c0392b">*** P&lt;{P_FLOOR or 0.001}</span> '
        '<span style="color:#e67e22">** P&lt;0.01</span> '
        '<span style="color:#f39c12">* P&lt;0.05</span>. '
        + star_note +
        'Every field starts at the development-cohort median or mode, so an untouched form describes a typical participant.</div>',
        unsafe_allow_html=True,
    )

    vals = {}
    for gi, (gname, feats) in enumerate(GROUPS):
        with st.expander(gname, expanded=(gi <= 1)):
            for f in feats:
                k = f["key"]                # 对外名称
                mk = mkey(k)                # 模型真实列名
                stars, colr = sig_stars(k)
                st.markdown(
                    f'**{f["label"]}** <span style="color:{colr};font-weight:700">{stars}</span>',
                    unsafe_allow_html=True,
                )
                if f["kind"] == "binary":
                    default_idx = int(DEFAULTS.get(mk, 0))
                    choice = st.selectbox(
                        f["label"], options=[0, 1],
                        format_func=lambda x, ff=f: ff["yes"] if x == 1 else ff["no"],
                        index=default_idx, help=f["help"] or hr_text(k),
                        label_visibility="collapsed", key=f"w_{k}",
                    )
                    vals[k] = float(choice)
                else:
                    lo, hi = RANGES.get(mk, [0.0, 1.0])
                    vals[k] = float(st.slider(
                        f["label"], min_value=float(lo), max_value=float(hi),
                        value=float(DEFAULTS.get(mk, (lo + hi) / 2)),
                        step=float(f["step"]), help=f["help"] or hr_text(k),
                        label_visibility="collapsed", key=f"w_{k}",
                    ))

        # education 放在 Demographics 之后
        if gname == "Demographics":
            with st.expander("Education", expanded=True):
                st.markdown("**Highest education level**", unsafe_allow_html=True)
                edu = st.selectbox(
                    "Education", options=list(EDU_OPTIONS.keys()), index=0,
                    label_visibility="collapsed", key="w_edu",
                    help="Reference category is level 1 (harmonised raeducl)",
                )
                vals["education_level2"], vals["education_level3"] = EDU_OPTIONS[edu]

    go = stretch(st.button, "🔍 Calculate risk", type="primary")


# =============================================================================
# 预测
# =============================================================================

def build_X(vals):
    """vals 用对外名称索引；这里还原成模型真实列名后再送 preprocessor。"""
    row = {}
    missing = []
    for c in DESIGN_ORDER:                  # c 是模型真实列名
        pub = pkey(c)                       # 界面上用的名字
        if pub in vals:
            row[c] = vals[pub]
        elif c in vals:                     # 容错：界面直接用了真实列名
            row[c] = vals[c]
        else:
            missing.append(pub)
    if missing:
        raise RuntimeError(f"Missing inputs for: {missing}")
    df_raw = pd.DataFrame([row])[DESIGN_ORDER]
    return preprocessor.transform(df_raw)


def surv_at(x, y, t):
    """在阶梯函数上取值：取最后一个 <= t 的时间点。"""
    if t < x[0]:
        return 1.0
    i = int(np.searchsorted(x, t, side="right")) - 1
    return float(y[i])


def _cumhaz(fn, grid):
    """累积风险阶梯函数取值。

    必须走 a/b 变换：sksurv 的 predict_cumulative_hazard_function 返回
    StepFunction(x=时点, y=**共享**基线, a=exp(线性预测子))，个体风险在 .a 里。
    直接读 .y 会让所有人拿到同一条基线曲线。
    """
    x = np.asarray(fn.x, float); y = np.asarray(fn.y, float)
    a = float(getattr(fn, "a", 1.0)); b = float(getattr(fn, "b", 0.0))
    return np.array([0.0 if t < x[0] else
                     a * y[int(np.searchsorted(x, t, side="right")) - 1] + b
                     for t in grid])


def compute_cif(X):
    """竞争风险下的累积发生率。

        CIF_1(t|x) = Σ_{u<=t} S(u-|x)·dH_1(u|x)
        S(t|x)     = Π_{u<=t} [1 - dH_1(u|x) - dH_2(u|x)]

    S 用乘积极限式而非 exp(-H)：认知在两年一次的调查波次上复测，事件时间
    高度并列、每个时点的 dH 很大，exp(-H) 会系统性高估 CIF。

    返回 (grid, cif_psci, cif_death)。
    """
    h1 = cox_model.predict_cumulative_hazard_function(X)[0]
    h2 = death_model.predict_cumulative_hazard_function(X)[0]
    grid = np.unique(np.concatenate([np.asarray(h1.x, float),
                                     np.asarray(h2.x, float)]))
    grid = grid[(grid > 0) & (grid <= 10.0)]
    H1 = _cumhaz(h1, grid)
    H2 = _cumhaz(h2, grid)
    dH1 = np.clip(np.diff(H1, prepend=0.0), 0.0, 1.0)
    dH2 = np.clip(np.diff(H2, prepend=0.0), 0.0, 1.0)
    S = np.cumprod(np.clip(1.0 - dH1 - dH2, 0.0, 1.0))
    S_prev = np.concatenate([[1.0], S[:-1]])
    return grid, np.cumsum(S_prev * dH1), np.cumsum(S_prev * dH2)


def curve_at(grid, vals, t):
    k = int(np.searchsorted(grid, t, side="right")) - 1
    return float(vals[k]) if k >= 0 else 0.0


def risk_label(p5):
    """按开发队列预测 5 年风险的三分位分级。

    旧的绝对阈值（3 年 <20%/20-40%/>=40%）不可用：开发队列 3 年 CIF 仅约 7.6%，
    几乎所有人都会落进 Low risk，分级没有区分度。
    """
    if not RISK_CUTS:
        return None, None
    if p5 >= RISK_CUTS["t67"]:
        return "Higher third", "risk-high"
    if p5 >= RISK_CUTS["t33"]:
        return "Middle third", "risk-medium"
    return "Lower third", "risk-low"


def plot_curve(grid, cif_psci, cif_death=None):
    """CIF 模式画 PSCI 与竞争死亡两条累积发生率；无死亡模型时退回 1-S(t)。"""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    r = np.asarray(cif_psci) * 100
    ax.step(grid, r, where="post", color="#e74c3c", lw=2.4,
            label="PSCI", zorder=3)
    ax.fill_between(grid, 0, r, step="post", alpha=0.12, color="#e74c3c")

    if cif_death is not None:
        d = np.asarray(cif_death) * 100
        ax.step(grid, d, where="post", color="#7f8c8d", lw=1.6, ls="--",
                label="Death before PSCI", zorder=2)

    for t in PRED_TIMES:
        if t > grid[-1]:
            continue
        v = curve_at(grid, cif_psci, t) * 100
        ax.axvline(t, color="#ddd", ls=":", lw=1, zorder=1)
        ax.scatter([t], [v], color="#c0392b", s=50, zorder=4)
        ax.annotate(f"{v:.1f}%", xy=(t, v), xytext=(t + 0.15, v + 1.8),
                    fontsize=9, color="#c0392b", fontweight="bold")

    ax.set_xlabel("Years since stroke", fontsize=10)
    ax.set_ylabel("Cumulative incidence (%)", fontsize=10)
    ax.set_title(
        "Cumulative incidence of PSCI" +
        (" (death as competing event)" if cif_death is not None
         else " (death treated as censoring)"),
        fontsize=10.5, fontweight="bold")
    top = max(r.max(), (np.asarray(cif_death).max() * 100) if cif_death is not None else 0)
    ax.set_xlim(0, max(grid[-1], max(PRED_TIMES)))
    ax.set_ylim(0, min(100, max(5.0, top * 1.2)))
    if cif_death is not None:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


def contributions(X):
    """每个预测因子相对于"队列平均患者"的 log HR 贡献。

    必须中心化：preprocessor 只对连续变量做了 StandardScaler，二分类未中心化。
    不减去队列均值的话，连续变量的参照是队列均值、二分类的参照却是取值 0，
    同一张图里两种参照混用，读者无法解释。
    减去均值后每根柱子统一表示"比平均患者高/低多少 log HR"，
    所有柱子之和 = 该患者的风险评分 − 队列平均风险评分。
    """
    names = list(preprocessor.get_feature_names_out())
    coefs = np.asarray(cox_model.coef_, dtype=float)
    xrow = np.asarray(X, dtype=float).ravel()
    out = {}
    for i, n in enumerate(names):
        bare = n.split("__", 1)[1] if "__" in n else n
        center = float(CENTERS.get(bare, 0.0))
        out[bare] = float(coefs[i] * (xrow[i] - center))
    return out


def plot_waterfall(contrib):
    contrib = dict(contrib)
    lab = {}
    # 同一概念预测因子的多个哑变量合成一根柱子（如教育的两个 level 哑变量）
    for merged_name, members in MERGE_BARS.items():
        present = [m for m in members if m in contrib]
        if len(present) >= 2:
            lab[merged_name] = sum(contrib.pop(m) for m in present)
    for k, v in contrib.items():          # k 是模型真实列名
        c = COEFS.get(k)
        lab[c["label"] if c else pkey(k)] = v
    items = sorted(lab.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    v = [x for _, x in items]
    colors = ["#e74c3c" if z > 0 else "#3498db" for z in v]

    fig, ax = plt.subplots(figsize=(6, max(5.5, len(labels) * 0.36)))
    bars = ax.barh(labels, v, color=colors, alpha=0.85, edgecolor="white")
    xm = max(abs(z) for z in v) if v else 0.1
    for b, z in zip(bars, v):
        off = xm * 0.03 if z >= 0 else -xm * 0.03
        ax.text(b.get_width() + off, b.get_y() + b.get_height() / 2, f"{z:+.3f}",
                va="center", ha="left" if z >= 0 else "right",
                fontsize=8, color="#333", fontweight="bold")
    ax.axvline(0, color="black", lw=1, alpha=0.6)
    ax.set_xlim(-xm * 1.35, xm * 1.35)
    ax.set_xlabel("Contribution to log hazard ratio\n(relative to the average patient)", fontsize=9)
    ax.set_title("Predictor contributions vs cohort average", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2, axis="x")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    return fig


with col_out:
    st.markdown("### Prediction")
    if not go:
        st.info("👈 Set the patient characteristics, then click **Calculate risk**.")
    else:
        try:
            X = build_X(vals)

            if CIF_MODE:
                grid, cif_p, cif_d = compute_cif(X)
                risks = {t: curve_at(grid, cif_p, t) for t in PRED_TIMES if t <= grid[-1]}
                death_risks = {t: curve_at(grid, cif_d, t) for t in risks}
                fig = plot_curve(grid, cif_p, cif_d)
                unit_label = "-year risk of PSCI"
            else:
                sf = cox_model.predict_survival_function(X)[0]
                x, y = np.asarray(sf.x, float), np.asarray(sf.y, float)
                grid = x
                cif_p = 1.0 - y
                risks = {t: 1 - surv_at(x, y, t) for t in PRED_TIMES if t <= x[-1]}
                death_risks = None
                fig = plot_curve(grid, cif_p, None)
                unit_label = "-year risk"

            cs = st.columns(len(risks))
            for i, (t, p) in enumerate(risks.items()):
                with cs[i]:
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-value">{p*100:.1f}%</div>'
                        f'<div class="metric-label">{t:.0f}{unit_label}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("")
            p5 = risks.get(5.0, list(risks.values())[-1])
            lbl, css = risk_label(p5)
            if lbl:
                st.markdown(
                    f'**Risk relative to the development cohort:** '
                    f'<span class="{css}">{lbl}</span>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="note-box">Based on the 5-year estimate against tertiles of '
                    f'predicted risk in the development cohort '
                    f'(cutpoints {RISK_CUTS["t33"]*100:.1f}% and {RISK_CUTS["t67"]*100:.1f}%). '
                    f'These are population reference points, not clinical action thresholds.</div>',
                    unsafe_allow_html=True)

            stretch(st.pyplot, fig)
            plt.close()

            if CIF_MODE:
                d10 = death_risks.get(10.0)
                death_txt = (f' For this patient the estimated probability of dying before '
                             f'PSCI within 10 years is {d10*100:.0f}%.' if d10 is not None else "")
                st.markdown(
                    '<div class="note-box"><b>How to read this.</b> These are cumulative '
                    '<i>incidences</i>: the probability of actually developing PSCI, with death '
                    f'before PSCI treated as a competing event rather than as censoring.{death_txt} '
                    'Cognition was reassessed at biennial survey waves, so the curve steps on '
                    'wave timing and incidence within the first ~2 years is not reliably '
                    'ascertained.'
                    '<br><br><b>Why a very frail patient can show a lower number.</b> Because '
                    'death competes with PSCI, cumulative incidence can be <i>lower</i> for a '
                    'patient at very high mortality risk than for an otherwise similar but '
                    'younger one. That is not because their cognition is more likely to be '
                    'spared — their rate of cognitive decline while alive is still higher — but '
                    'because they are less likely to survive long enough for PSCI to occur.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="warn-box"><b>Interpretation caveat.</b> These are '
                    '<i>cause-specific</i> risks estimated with death treated as censoring. '
                    f'Death before PSCI occurred in {COH.get("pct_death_before_psci","~25")}% of '
                    'the development cohort, so the 10-year value <b>overestimates</b> the '
                    'absolute probability of developing PSCI. Run step_E_competing_risk.py and '
                    're-export the bundle to switch this calculator to cumulative incidence.</div>',
                    unsafe_allow_html=True)

            st.session_state["wf"] = plot_waterfall(contributions(X))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col_wf:
    st.markdown("### Predictor contributions")
    if go and "wf" in st.session_state:
        stretch(st.pyplot, st.session_state["wf"])
        plt.close()
        st.markdown(
            '<div class="note-box"><b>Red</b>: increases predicted risk | '
            '<b>Blue</b>: decreases it. Each bar is measured <b>against the average '
            'participant in the development cohort</b>, so a bar near zero means this patient '
            'resembles the cohort average on that predictor; the bars sum to this patient\'s '
            'risk score minus the cohort-average risk score.<br>'
            'Coefficients are predictive, <b>not causal</b>: several classic vascular risk factors '
            'point in a protective direction in this stroke-survivor population because of '
            'index event bias.</div>',
            unsafe_allow_html=True)
    else:
        st.info("Contributions appear here after you calculate a risk.")

# =============================================================================
# 页脚
# =============================================================================

st.divider()
perf_bits = []
for key, name in [("internal", "Internal"), ("external_share", "External (SHARE)")]:
    p = PERF.get(key)
    if p:
        perf_bits.append(f"{name} C-index {p['c_index']:.3f} (95% CI {p['lo']:.3f}–{p['hi']:.3f}; n = {p['n']:,})")
perf_line = "; ".join(perf_bits) if perf_bits else "Performance metrics unavailable."

st.markdown(f"""
<div style="font-size:0.8rem;color:#888;line-height:1.6">
<b>Model.</b> Cox proportional-hazards model with 17 prestroke predictors (18 design columns).
Discrimination is reported for the primary model fitted on the 80% training partition:
{perf_line} The calculator uses coefficients refitted on the full development set
({COH.get('n_development','?'):,} participants; {COH.get('n_psci','?'):,} PSCI events,
{COH.get('n_death_before_psci','?'):,} deaths before PSCI) to maximise precision; no performance
metric is recomputed from that refit.
{"<br><br><b>Competing risk.</b> Cumulative incidence is obtained by combining cause-specific Cox models for PSCI and for death, using the product-integral form. A Fine-Gray sensitivity analysis gave subdistribution hazard ratios consistent in direction with the cause-specific model for 17 of 18 coefficients." if CIF_MODE else ""}
<br><br>
<b>Interpretation.</b> Coefficients are predictive, not causal. In this stroke-survivor
population several classic vascular risk factors point in a protective direction, which is
consistent with index event bias rather than a protective biological effect.
<br><br>
<b>Disclaimer.</b> Research use only. Not a substitute for clinical judgement. The model was
developed in community-dwelling cohorts and may not transfer to hospital stroke populations,
acute stroke severity, or imaging-defined subtypes.
</div>
""", unsafe_allow_html=True)
