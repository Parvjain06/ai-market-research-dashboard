import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from wordcloud import WordCloud
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

st.set_page_config(
    page_title="SentIQ — Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ivory:  #FDFAF4;
    --linen:  #F5F0E8;
    --sage:   #B8B5A8;
    --taupe:  #6B6560;
    --ink:    #2C2825;
    --green:  #4A7C59;
    --red:    #B85450;
    --amber:  #C9973A;
    --border: rgba(184,181,168,0.55);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--ivory) !important;
    color: var(--ink) !important;
}
.main .block-container {
    padding: 0 2.5rem 2rem 2.5rem !important;
    max-width: 1500px;
    background: var(--ivory);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ── PAGE HEADER ── */
.page-header { margin-bottom: 1.6rem; }
.page-title {
    font-family: 'Lora', serif;
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--ink);
    margin: 0;
    letter-spacing: -0.01em;
}
.page-subtitle {
    font-size: 0.84rem;
    color: var(--taupe);
    margin-top: 0.25rem;
    font-weight: 300;
}

/* ── FILTER CARD — applied via CSS to the container ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--linen) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.2rem 0.8rem 0.6rem !important;
    margin-bottom: 1.4rem !important;
}

/* filter widget labels */
div[data-testid="stVerticalBlockBorderWrapper"] label p,
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] p {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #B8B5A8 !important;
    -webkit-text-fill-color: #B8B5A8 !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* filter multiselect boxes */
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMultiSelect"] > div {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(107,101,96,0.12) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}

/* filter text input */
div[data-testid="stVerticalBlockBorderWrapper"] input {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
}

/* filter selectbox */
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] > div > div {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
}

/* filter slider */
div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--taupe) !important;
    border-color: var(--taupe) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-baseweb="slider"] > div > div > div:nth-child(2) {
    background: var(--taupe) !important;
}

/* ── KPI CARDS ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 12px;
    margin-bottom: 1.6rem;
}
.kpi-card {
    background: var(--linen);
    border: 1px solid var(--border);
    border-top: 3px solid var(--sage);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
}
.kpi-card.green { border-top-color: var(--green); }
.kpi-card.red   { border-top-color: var(--red); }
.kpi-card.amber { border-top-color: var(--amber); }
.kpi-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6B6560 !important;
    -webkit-text-fill-color: #6B6560 !important;
    font-weight: 500;
}
.kpi-value {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 500;
    line-height: 1.1;
    margin: 0.3rem 0 0.15rem;
    color: #2C2825 !important;
    -webkit-text-fill-color: #2C2825 !important;
}
.kpi-card.green .kpi-value { color: #4A7C59 !important; -webkit-text-fill-color: #4A7C59 !important; }
.kpi-card.red   .kpi-value { color: #B85450 !important; -webkit-text-fill-color: #B85450 !important; }
.kpi-card.amber .kpi-value { color: #C9973A !important; -webkit-text-fill-color: #C9973A !important; }
.kpi-sub {
    font-size: 0.76rem;
    color: #B8B5A8 !important;
    -webkit-text-fill-color: #B8B5A8 !important;
}

/* ── INNER TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(184,181,168,0.55) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 4px !important;
    margin-bottom: 1.4rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #B8B5A8 !important;
    -webkit-text-fill-color: #B8B5A8 !important;
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.55rem 1.4rem !important;
    margin-bottom: -1px !important;
    outline: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2C2825 !important;
    -webkit-text-fill-color: #2C2825 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2C2825 !important;
    -webkit-text-fill-color: #2C2825 !important;
    font-weight: 500 !important;
    border: none !important;
    border-bottom: 2px solid #2C2825 !important;
    margin-bottom: -1px !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    background: transparent !important;
}

/* ── SECTION CARD ── */
.section-card {
    background: var(--linen);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--taupe);
    font-weight: 500;
    margin-bottom: 0.9rem;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--ivory) !important;
    -webkit-text-fill-color: var(--ivory) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── INPUTS ── */
.stTextInput input, .stTextArea textarea {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--taupe) !important;
    box-shadow: none !important;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    font-size: 0.87rem !important;
}

/* ── MULTISELECT ── */
[data-testid="stMultiSelect"] > div {
    background: var(--ivory) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(107,101,96,0.13) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 5px !important;
}

/* ── GLOBAL LABELS ── */
div[data-testid="stMultiSelect"] label p,
div[data-testid="stTextInput"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stSlider"] label p {
    font-size: 0.72rem !important;
    color: var(--taupe) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* ── PROGRESS ── */
.stProgress > div > div { background: var(--taupe) !important; border-radius: 6px !important; }
.stProgress > div { background: rgba(184,181,168,0.25) !important; border-radius: 6px !important; }

/* ── ALERT ── */
div[data-testid="stAlert"] {
    background: rgba(184,181,168,0.15) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span { color: var(--taupe) !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--linen) !important;
    border: 1px dashed var(--sage) !important;
    border-radius: 10px !important;
}

/* ── SLIDER ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--taupe) !important; border-color: var(--taupe) !important; }
div[data-baseweb="slider"] > div > div > div:nth-child(2) { background: var(--taupe) !important; }
.stSlider p { color: var(--taupe) !important; font-size: 0.78rem !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden; }

/* ── DIVIDER ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.1rem 0 !important; }

/* ── PILLS ── */
.pill { display: inline-block; font-size: 0.74rem; padding: 2px 9px; border-radius: 20px; font-weight: 500; }
.pill-pos { background: rgba(74,124,89,0.1);  color: var(--green); border: 1px solid rgba(74,124,89,0.25); }
.pill-neg { background: rgba(184,84,80,0.1);  color: var(--red);   border: 1px solid rgba(184,84,80,0.25); }
.pill-neu { background: rgba(201,151,58,0.1); color: var(--amber); border: 1px solid rgba(201,151,58,0.25); }

/* ── AI CHAT ── */
.chat-msg-user {
    background: rgba(107,101,96,0.09); border: 1px solid var(--border);
    border-radius: 10px 10px 2px 10px; padding: 0.65rem 0.9rem;
    margin: 0.4rem 0; font-size: 0.85rem; color: var(--ink); line-height: 1.5;
}
.chat-msg-ai {
    background: var(--linen); border: 1px solid var(--border);
    border-left: 3px solid var(--taupe); border-radius: 2px 10px 10px 10px;
    padding: 0.65rem 0.9rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--ink); line-height: 1.6;
}
.chat-who { font-size: 0.67rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.2rem; }
.chat-who-you { color: var(--taupe); }
.chat-who-ai  { color: var(--green); }

/* ── INSIGHT ── */
.insight {
    background: var(--ivory); border: 1px solid var(--border);
    border-left: 3px solid var(--taupe); border-radius: 0 8px 8px 0;
    padding: 0.65rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.85rem; color: var(--ink); line-height: 1.5;
}
.insight.warn { border-left-color: var(--red); }
.insight.info { border-left-color: var(--amber); }

/* ── SCORE BARS ── */
.sbar-wrap { margin: 0.35rem 0; }
.sbar-top  { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--taupe); margin-bottom: 3px; }
.sbar-track { height: 5px; background: rgba(184,181,168,0.28); border-radius: 3px; }
.sbar-fill  { height: 5px; border-radius: 3px; }

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 5rem 2rem; color: var(--taupe); }
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; }
.empty-title { font-family: 'Lora', serif; font-size: 1.2rem; font-weight: 500; color: var(--ink); margin-bottom: 0.4rem; }
.empty-sub   { font-size: 0.87rem; line-height: 1.6; }
code { background: rgba(184,181,168,0.2); color: var(--taupe); padding: 1px 5px; border-radius: 4px; font-size: 0.84em; }
</style>
""", unsafe_allow_html=True)


# ── DEVICE ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")

DEVICE          = get_device()
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
MODEL_PATH      = "parvj-06/roberta-sentiment-classweighted"
ASPECTS         = ["price","quality","delivery","service","packaging","quantity"]
ASPECT_ICONS    = {"price":"💰","quality":"⭐","delivery":"🚚","service":"🎧","packaging":"📦","quantity":"⚖️"}
COLORS          = {"positive":"#4A7C59","negative":"#B85450","neutral":"#C9973A"}
PLOTLY_LAYOUT   = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#6B6560", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(245,240,232,0.9)", bordercolor="rgba(184,181,168,0.6)",
                borderwidth=1, font=dict(color="#2C2825", size=11))
)


# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    mdl = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    mdl.eval()
    le_path = hf_hub_download(repo_id=MODEL_PATH, filename="label_encoder.pkl")
    le = joblib.load(le_path)
    absa = pipeline(
        task="text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        tokenizer="yangheng/deberta-v3-base-absa-v1.1",
        device=PIPELINE_DEVICE, truncation=True
    )
    return tokenizer, mdl, le, absa

with st.spinner("Loading models…"):
    tokenizer, model, le, absa_pipeline = load_models()


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def roberta_predict_batch(texts, batch_size=32):
    all_labels, all_confs = [], []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        idxs  = np.argmax(probs, axis=1)
        all_labels.extend(le.inverse_transform(idxs))
        all_confs.extend(probs[np.arange(len(batch)), idxs].tolist())
    return all_labels, all_confs

def extract_absa_batch(texts, batch_size=32):
    pairs   = [f"{asp}: {t}" for t in texts for asp in ASPECTS]
    outputs = absa_pipeline(pairs, truncation=True, max_length=256, batch_size=batch_size)
    results = []
    for i in range(len(texts)):
        for j, asp in enumerate(ASPECTS):
            results.append((asp, outputs[i*len(ASPECTS)+j]["label"].lower()))
    return results


# ── GROQ AI ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are SentIQ, a precise market research analyst. You have REAL data from customer review analysis.

STRICT RULES:
1. Only use the data provided. Never invent statistics.
2. Always quote actual numbers and percentages.
3. Be concise and structured — short paragraphs or bullets.
4. End with 1-2 concrete recommendations grounded in the data.
5. If data is insufficient, say so honestly."""

def call_groq(prompt):
    try:
        client   = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            max_tokens=1200, temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def build_context(df_f, aspect_df):
    total    = len(df_f)
    sent     = df_f["predicted_sentiment"].value_counts()
    avg_c    = df_f["confidence"].mean()
    neg_asp  = aspect_df[aspect_df["Sentiment"]=="negative"]["Aspect"].value_counts()
    pos_asp  = aspect_df[aspect_df["Sentiment"]=="positive"]["Aspect"].value_counts()
    neg_revs = df_f[df_f["predicted_sentiment"]=="negative"]["review"].head(8).tolist()
    pos_revs = df_f[df_f["predicted_sentiment"]=="positive"]["review"].head(3).tolist()
    words    = re.findall(r'\b[a-z]{4,}\b', " ".join(neg_revs).lower())
    stops    = {"this","that","with","have","from","they","were","been","their","what","when","just","also","very","more","than","some","would","could","product","item","like","good","great","really","about","even","dont","never","still","after"}
    top_kw   = Counter(w for w in words if w not in stops).most_common(10)
    prod_bk  = df_f.groupby("product")["predicted_sentiment"].value_counts().to_string() if "product" in df_f.columns else "N/A"
    return f"""DATASET: {total} reviews | Avg confidence: {avg_c:.1%}

SENTIMENT:
{sent.to_string()}
Positive {sent.get('positive',0)/total*100:.1f}% | Negative {sent.get('negative',0)/total*100:.1f}% | Neutral {sent.get('neutral',0)/total*100:.1f}%

NEGATIVE ASPECTS: {neg_asp.to_string() if not neg_asp.empty else 'None'}
POSITIVE ASPECTS: {pos_asp.to_string() if not pos_asp.empty else 'None'}
TOP NEGATIVE KEYWORDS: {', '.join([f"{w}({c})" for w,c in top_kw])}

SAMPLE NEGATIVE REVIEWS:
{chr(10).join([f"- {r}" for r in neg_revs])}

SAMPLE POSITIVE REVIEWS:
{chr(10).join([f"- {r}" for r in pos_revs])}

BY PRODUCT:
{prod_bk}""".strip()


# ── TOP NAV ───────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "analyze"

pages    = ["📊  Upload & Analyze", "🔍  Single Review", "📈  Model Comparison"]
page_map = {"📊  Upload & Analyze": "analyze", "🔍  Single Review": "single", "📈  Model Comparison": "models"}
rmap     = {v: k for k, v in page_map.items()}

def _tab(label, key, current):
    s = ("color:#2C2825;font-weight:500;border-bottom:2px solid #2C2825;"
         if current == key else "color:#B8B5A8;border-bottom:2px solid transparent;")
    return (f'<a href="?p={key}" target="_self" style="text-decoration:none;'
            f'padding:0 0 0.6rem 0;margin-left:2.2rem;font-size:0.9rem;'
            f'font-family:DM Sans,sans-serif;cursor:pointer;{s}">{label}</a>')

st.markdown(f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;
            border-bottom:1px solid rgba(184,181,168,0.55);
            padding-bottom:0;margin-bottom:1.8rem;">
    <div style="font-family:Lora,serif;font-size:2rem;font-weight:500;
                color:#2C2825;padding-bottom:0.6rem;">
        Sent<span style="color:#4A7C59;">IQ</span>
    </div>
    <div style="display:flex;align-items:flex-end;padding-bottom:0;margin-bottom:-1px;">
        {_tab("📊&nbsp;&nbsp;Upload &amp; Analyze","analyze",st.session_state.page)}
        {_tab("🔍&nbsp;&nbsp;Single Review","single",st.session_state.page)}
        {_tab("📈&nbsp;&nbsp;Model Comparison","models",st.session_state.page)}
    </div>
</div>
""", unsafe_allow_html=True)

_p = st.query_params.get("p", "")
if _p in ("analyze","single","models") and _p != st.session_state.page:
    st.session_state.page = _p
    st.rerun()

page = st.session_state.page


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD & ANALYZE
# ════════════════════════════════════════════════════════════════════════════
if page == "analyze":

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Market Intelligence Dashboard</div>
        <div class='page-subtitle'>Sentiment analysis · Aspect breakdown · AI insights</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column."); st.stop()
        if "date"    in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "product" not in df.columns: df["product"] = "General"

        if st.session_state.get("last_file") != uploaded_file.name:
            st.session_state.last_file     = uploaded_file.name
            st.session_state.analysis_done = False
            st.session_state.ai_chat       = []

        if not st.session_state.get("analysis_done", False):
            texts = df["review"].astype(str).tolist()
            prog  = st.progress(0, text="Running sentiment analysis…")
            sents, confs = roberta_predict_batch(texts, batch_size=32)
            prog.progress(55, text="Running aspect analysis…")
            asp_results  = extract_absa_batch(texts, batch_size=32)
            prog.progress(100, text="Done.")
            prog.empty()
            df["predicted_sentiment"] = sents
            df["confidence"]          = confs
            df["review_length"]       = df["review"].str.len()
            st.session_state.df        = df
            st.session_state.aspect_df = pd.DataFrame(asp_results, columns=["Aspect","Sentiment"])
            st.session_state.analysis_done = True
            st.success(f"Analysed {len(df)} reviews.")

        df        = st.session_state.df
        aspect_df = st.session_state.aspect_df

        # ── FILTER BAR ─────────────────────────────────────────────────────
        # Use st.container(border=True) — this renders a real bordered box
        # that Streamlit widgets naturally sit inside.
        with st.container(border=True):
            fc1, fc2, fc3, fc4, fc5 = st.columns([2.8, 2.2, 1.8, 1.4, 1.4])

            with fc1:
                prod_filter = st.multiselect("Product", df["product"].unique(), df["product"].unique())
            df_f = df[df["product"].isin(prod_filter)]

            with fc2:
                sent_opts   = [s for s in ["positive","negative","neutral"] if s in df_f["predicted_sentiment"].unique()]
                sent_filter = st.multiselect("Sentiment", sent_opts, sent_opts)
                df_f = df_f[df_f["predicted_sentiment"].isin(sent_filter)] if sent_filter else df_f

            with fc3:
                kw = st.text_input("Keyword", placeholder="Search…")
                if kw: df_f = df_f[df_f["review"].str.contains(kw, case=False, na=False)]

            with fc4:
                conf_thr = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
                df_f = df_f[df_f["confidence"] >= conf_thr]

            with fc5:
                top_n = st.selectbox("Show", [50, 100, 200, 500, "All"], index=1)
                if top_n != "All": df_f = df_f.head(int(top_n))

            # second row: date + count
            r1, r2 = st.columns([3, 6])
            with r1:
                if "date" in df_f.columns and df_f["date"].notna().any():
                    dr = st.date_input("Date range", [df_f["date"].min(), df_f["date"].max()],
                                       label_visibility="visible")
                    if len(dr) == 2:
                        df_f = df_f[(df_f["date"]>=pd.to_datetime(dr[0]))&(df_f["date"]<=pd.to_datetime(dr[1]))]
            with r2:
                st.markdown(f"""
                <div style='padding-top:1.6rem;font-size:0.8rem;color:#B8B5A8;'>
                    Showing <strong style='color:#2C2825;'>{len(df_f)}</strong> of {len(df)} reviews
                </div>""", unsafe_allow_html=True)

        # ── KPIS ───────────────────────────────────────────────────────────
        total  = max(len(df_f), 1)
        n_pos  = (df_f["predicted_sentiment"]=="positive").sum()
        n_neg  = (df_f["predicted_sentiment"]=="negative").sum()
        avg_cf = df_f["confidence"].mean()*100

        st.markdown(f"""
        <div class='kpi-row'>
            <div class='kpi-card'>
                <div class='kpi-label'>Total Reviews</div>
                <div class='kpi-value'>{total}</div>
                <div class='kpi-sub'>after filters</div>
            </div>
            <div class='kpi-card green'>
                <div class='kpi-label'>Positive</div>
                <div class='kpi-value'>{n_pos/total*100:.0f}%</div>
                <div class='kpi-sub'>{n_pos} reviews</div>
            </div>
            <div class='kpi-card red'>
                <div class='kpi-label'>Negative</div>
                <div class='kpi-value'>{n_neg/total*100:.0f}%</div>
                <div class='kpi-sub'>{n_neg} reviews</div>
            </div>
            <div class='kpi-card amber'>
                <div class='kpi-label'>Avg Confidence</div>
                <div class='kpi-value'>{avg_cf:.0f}%</div>
                <div class='kpi-sub'>model certainty</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MAIN + AI COLUMNS ──────────────────────────────────────────────
        main_col, ai_col = st.columns([3, 1])

        with main_col:
            tabs = st.tabs(["📊  Overview", "🔍  Aspects", "📈  Trends", "☁️  Word Cloud", "🗂  Data"])

            with tabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    counts  = df_f["predicted_sentiment"].value_counts()
                    fig_pie = go.Figure(go.Pie(
                        labels=counts.index, values=counts.values, hole=0.62,
                        marker=dict(colors=[COLORS.get(l,"#B8B5A8") for l in counts.index],
                                    line=dict(color="#FDFAF4", width=3)),
                        textfont=dict(family="DM Sans", size=12, color="#2C2825"),
                        hovertemplate="<b>%{label}</b><br>%{value} reviews (%{percent})<extra></extra>"
                    ))
                    fig_pie.add_annotation(text=f"<b>{total}</b>", x=0.5, y=0.5, showarrow=False,
                                           font=dict(color="#2C2825", size=20, family="Lora"))
                    fig_pie.update_layout(
                        title=dict(text="Sentiment Distribution", font=dict(family="Lora", size=14, color="#2C2825")),
                        **PLOTLY_LAYOUT, height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    cdf   = df_f.groupby("predicted_sentiment")["confidence"].mean().reset_index()
                    fig_c = go.Figure(go.Bar(
                        x=cdf["predicted_sentiment"], y=cdf["confidence"],
                        marker=dict(color=[COLORS.get(l,"#B8B5A8") for l in cdf["predicted_sentiment"]], line=dict(width=0)),
                        text=[f"{v:.1%}" for v in cdf["confidence"]], textposition="outside",
                        textfont=dict(color="#2C2825", size=12, family="DM Sans")
                    ))
                    fig_c.update_layout(
                        title=dict(text="Avg Confidence by Sentiment", font=dict(family="Lora", size=14, color="#2C2825")),
                        yaxis=dict(tickformat=".0%", gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                        xaxis=dict(color="#6B6560"), **PLOTLY_LAYOUT, height=300)
                    st.plotly_chart(fig_c, use_container_width=True)

                fig_h = px.histogram(df_f, x="confidence", color="predicted_sentiment",
                                     color_discrete_map=COLORS, nbins=30, barmode="overlay", opacity=0.7,
                                     title="Confidence Score Distribution")
                fig_h.update_layout(**PLOTLY_LAYOUT, height=240,
                                    title=dict(font=dict(family="Lora", size=14, color="#2C2825")),
                                    xaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                                    yaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"))
                st.plotly_chart(fig_h, use_container_width=True)

            with tabs[1]:
                if not aspect_df.empty:
                    pivot = aspect_df.value_counts().reset_index(name="Count")
                    pw    = pivot.pivot(index="Aspect", columns="Sentiment", values="Count").fillna(0).reset_index()
                    fig_asp = go.Figure()
                    for s in ["positive","negative","neutral"]:
                        if s in pw.columns:
                            fig_asp.add_trace(go.Bar(name=s.capitalize(), x=pw["Aspect"], y=pw[s],
                                                     marker_color=COLORS[s], marker_line_width=0,
                                                     hovertemplate=f"<b>%{{x}}</b> — {s}<br>%{{y}}<extra></extra>"))
                    fig_asp.update_layout(barmode="group",
                        title=dict(text="Aspect-Level Sentiment", font=dict(family="Lora", size=14, color="#2C2825")),
                        xaxis=dict(color="#6B6560"), yaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                        **PLOTLY_LAYOUT, height=340)
                    st.plotly_chart(fig_asp, use_container_width=True)

                    pos_s  = aspect_df[aspect_df["Sentiment"]=="positive"].groupby("Aspect").size()
                    tot_s  = aspect_df.groupby("Aspect").size()
                    health = (pos_s / tot_s * 100).fillna(0)
                    fig_r  = go.Figure(go.Scatterpolar(
                        r=[health.get(a,0) for a in ASPECTS],
                        theta=[f"{ASPECT_ICONS[a]} {a.capitalize()}" for a in ASPECTS],
                        fill='toself', fillcolor='rgba(74,124,89,0.1)',
                        line=dict(color='#4A7C59', width=2), marker=dict(color='#4A7C59', size=7)
                    ))
                    fig_r.update_layout(
                        polar=dict(bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(184,181,168,0.4)",
                                            color="#6B6560", ticksuffix="%"),
                            angularaxis=dict(gridcolor="rgba(184,181,168,0.4)", color="#2C2825")),
                        title=dict(text="Aspect Health (% Positive)", font=dict(family="Lora", size=14, color="#2C2825")),
                        **PLOTLY_LAYOUT, height=360)
                    st.plotly_chart(fig_r, use_container_width=True)

                    st.markdown("<div class='section-label'>⚠ Problem areas</div>", unsafe_allow_html=True)
                    for asp, cnt in aspect_df[aspect_df["Sentiment"]=="negative"]["Aspect"].value_counts().items():
                        pct = cnt/total*100
                        st.markdown(f"""
                        <div class='sbar-wrap'>
                            <div class='sbar-top'>
                                <span>{ASPECT_ICONS.get(asp,'·')} {asp.capitalize()}</span>
                                <span style='color:var(--red);'>{cnt} complaints ({pct:.1f}%)</span>
                            </div>
                            <div class='sbar-track'>
                                <div class='sbar-fill' style='width:{min(pct*2,100):.0f}%;background:var(--red);opacity:0.65;'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

            with tabs[2]:
                if "date" in df_f.columns and df_f["date"].notna().any():
                    trend = df_f.groupby([df_f["date"].dt.to_period("M"),"predicted_sentiment"]).size().unstack().fillna(0)
                    trend.index = trend.index.astype(str)
                    fig_t = go.Figure()
                    for s in trend.columns:
                        fig_t.add_trace(go.Scatter(x=trend.index, y=trend[s], name=s.capitalize(),
                            mode="lines+markers", line=dict(color=COLORS.get(s,"#B8B5A8"), width=2),
                            marker=dict(size=6, color=COLORS.get(s,"#B8B5A8"))))
                    fig_t.update_layout(
                        title=dict(text="Sentiment Over Time", font=dict(family="Lora", size=14, color="#2C2825")),
                        xaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                        yaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                        **PLOTLY_LAYOUT, height=340)
                    st.plotly_chart(fig_t, use_container_width=True)

                    df_s = df_f.sort_values("date").copy()
                    df_s["score"]   = df_s["predicted_sentiment"].map({"positive":1,"neutral":0,"negative":-1})
                    df_s["rolling"] = df_s["score"].rolling(window=min(20,len(df_s))).mean()
                    fig_roll = px.line(df_s, x="date", y="rolling", title="Rolling Sentiment Score",
                                       color_discrete_sequence=["#6B6560"])
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="#B85450", opacity=0.4)
                    fig_roll.update_layout(**PLOTLY_LAYOUT, height=240,
                        title=dict(font=dict(family="Lora", size=14, color="#2C2825")),
                        xaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"),
                        yaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560"))
                    st.plotly_chart(fig_roll, use_container_width=True)
                else:
                    st.info("Add a 'date' column to your CSV to unlock trend analysis.")

            with tabs[3]:
                wc_s    = st.radio("Generate for:", ["negative","positive","neutral"], horizontal=True)
                wc_text = " ".join(df_f[df_f["predicted_sentiment"]==wc_s]["review"].astype(str))
                if wc_text.strip():
                    cmap  = {"positive":"YlGn","negative":"OrRd","neutral":"Greys"}[wc_s]
                    wc    = WordCloud(width=900, height=400, background_color="#F5F0E8",
                                      colormap=cmap, max_words=80).generate(wc_text)
                    fig_wc, ax = plt.subplots(figsize=(11,4))
                    fig_wc.patch.set_facecolor("#F5F0E8"); ax.set_facecolor("#F5F0E8")
                    ax.imshow(wc); ax.axis("off"); plt.tight_layout(pad=0)
                    st.pyplot(fig_wc)
                else:
                    st.info(f"No {wc_s} reviews found.")

            with tabs[4]:
                st.markdown(f"<div class='section-label'>{len(df_f)} reviews</div>", unsafe_allow_html=True)
                display_cols = ["review","predicted_sentiment","confidence","product"] + (["date"] if "date" in df_f.columns else [])
                st.dataframe(df_f[display_cols], use_container_width=True, height=400)
                c1, c2 = st.columns(2)
                with c1: st.download_button("Download filtered CSV", df_f.to_csv(index=False).encode(), "filtered.csv","text/csv")
                with c2: st.download_button("Download full CSV",     df.to_csv(index=False).encode(),   "full.csv","text/csv")

        # ── AI PANEL ──────────────────────────────────────────────────────
        with ai_col:
            st.markdown("""
            <div style='border-bottom:1px solid rgba(184,181,168,0.55);padding-bottom:0.75rem;margin-bottom:0.75rem;'>
                <div style='font-family:Lora,serif;font-size:0.95rem;font-weight:500;color:#2C2825;'>SentIQ Analyst</div>
                <div style='font-size:0.71rem;color:#B8B5A8;margin-top:0.1rem;'>Answers grounded in your data</div>
            </div>
            """, unsafe_allow_html=True)

            if "ai_chat" not in st.session_state:
                st.session_state.ai_chat = []

            chat_box = st.container(height=340)
            with chat_box:
                if not st.session_state.ai_chat:
                    st.markdown("<div style='padding:1.5rem 0.5rem;color:#B8B5A8;font-size:0.83rem;text-align:center;'>Ask me anything about your review data.</div>", unsafe_allow_html=True)
                for role, msg in st.session_state.ai_chat:
                    if role == "user":
                        st.markdown(f"<div class='chat-who chat-who-you'>You</div><div class='chat-msg-user'>{msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-who chat-who-ai'>Analyst</div><div class='chat-msg-ai'>{msg}</div>", unsafe_allow_html=True)

            user_q = st.text_input("Ask…", key="uq", label_visibility="collapsed", placeholder="e.g. What are the main complaints?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Ask", use_container_width=True):
                    if user_q.strip():
                        ctx = build_context(df_f, aspect_df)
                        with st.spinner("Thinking…"):
                            reply = call_groq(f"DATA:\n{ctx}\n\nQUESTION: {user_q}")
                        st.session_state.ai_chat.extend([("user",user_q),("ai",reply)])
                        st.rerun()
            with c2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.ai_chat = []; st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.09em;color:#B8B5A8;margin-bottom:0.5rem;'>Quick insights</div>", unsafe_allow_html=True)

            for lbl, qp in {
                "Top issues":        "What are the top 3 critical issues? Be specific with numbers.",
                "Worst aspect":      "Which aspect has the worst sentiment and what do customers say?",
                "Recommendations":   "Give 3 concrete business recommendations based strictly on this data.",
                "Executive summary": "Give a 5-bullet executive summary with exact numbers."
            }.items():
                if st.button(lbl, use_container_width=True, key=f"q_{lbl}"):
                    ctx = build_context(df_f, aspect_df)
                    with st.spinner("Thinking…"):
                        reply = call_groq(f"DATA:\n{ctx}\n\nQUESTION: {qp}")
                    st.session_state.ai_chat.extend([("user",lbl),("ai",reply)])
                    st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)
            if st.button("Generate full report", use_container_width=True):
                ctx = build_context(df_f, aspect_df)
                rp  = f"DATA:\n{ctx}\n\nWrite a structured report:\n1. Executive Summary (3 sentences, include %s)\n2. Critical Problems (top 3 with evidence)\n3. Aspect-wise Issues\n4. 5 Actionable Recommendations\n5. Single most important action"
                with st.spinner("Generating…"):
                    report = call_groq(rp)
                st.download_button("Download report", report, "report.txt","text/plain")

            if st.session_state.ai_chat:
                chat_txt = "\n\n".join([f"{r.upper()}: {m}" for r,m in st.session_state.ai_chat])
                st.download_button("Download chat", chat_txt, "chat.txt","text/plain")

    else:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>📂</div>
            <div class='empty-title'>Upload your review dataset</div>
            <div class='empty-sub'>CSV with a <code>review</code> column required<br>Optional: <code>product</code> · <code>date</code> · <code>rating</code></div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: SINGLE REVIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "single":

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Single Review Analyser</div>
        <div class='page-subtitle'>Instant sentiment and aspect breakdown for any review</div>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area("Paste a review:", height=130,
                        placeholder="e.g. Quality is great but delivery took 3 weeks and packaging was damaged…")
    if st.button("Analyse"):
        if text.strip():
            with st.spinner("Analysing…"):
                labels, confs = roberta_predict_batch([text])
                absa_results  = extract_absa_batch([text])
            sentiment, confidence = labels[0], confs[0]
            color = COLORS.get(sentiment, "#B8B5A8")
            st.markdown(f"""
            <div style='background:var(--linen);border:1px solid rgba(184,181,168,0.6);border-left:4px solid {color};
                        border-radius:10px;padding:1.3rem 1.5rem;margin:1rem 0;'>
                <div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#B8B5A8;'>Overall sentiment</div>
                <div style='font-family:Lora,serif;font-size:2.2rem;font-weight:500;color:{color};margin:0.25rem 0 0.2rem;'>{sentiment.capitalize()}</div>
                <div style='font-size:0.84rem;color:#6B6560;'>Confidence: <strong style='color:{color};'>{confidence:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='section-label' style='margin-top:1.2rem;'>Aspect breakdown</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (asp, sent) in enumerate(absa_results):
                c = COLORS.get(sent, "#B8B5A8")
                with cols[i%3]:
                    st.markdown(f"""
                    <div style='background:var(--linen);border:1px solid rgba(184,181,168,0.6);border-top:2px solid {c};
                                border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.6rem;'>
                        <div style='font-size:1rem;'>{ASPECT_ICONS.get(asp,'·')}</div>
                        <div style='font-size:0.84rem;color:#2C2825;font-weight:500;margin:0.2rem 0 0.3rem;'>{asp.capitalize()}</div>
                        <span class='pill pill-{"pos" if sent=="positive" else "neg" if sent=="negative" else "neu"}'>{sent}</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "models":

    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Model Performance</div>
        <div class='page-subtitle'>SVM · BERT · RoBERTa — identical training splits, class-weighted</div>
    </div>
    """, unsafe_allow_html=True)

    data = {"Model":["SVM","BERT","RoBERTa"],
            "Accuracy":[0.8251,0.8321,0.8763],"Precision":[0.7912,0.8578,0.8851],
            "Recall":[0.8251,0.8321,0.8763],"F1 Score":[0.7948,0.8432,0.8803],"F1 Macro":[0.5523,0.6653,0.7166]}
    df_m = pd.DataFrame(data)

    st.markdown("""
    <div class='kpi-row'>
        <div class='kpi-card green'><div class='kpi-label'>Best accuracy</div><div class='kpi-value'>87.6%</div><div class='kpi-sub'>RoBERTa</div></div>
        <div class='kpi-card green'><div class='kpi-label'>Best F1</div><div class='kpi-value'>88.0%</div><div class='kpi-sub'>RoBERTa</div></div>
        <div class='kpi-card'><div class='kpi-label'>SVM baseline</div><div class='kpi-value'>82.5%</div><div class='kpi-sub'>Accuracy</div></div>
        <div class='kpi-card amber'><div class='kpi-label'>RoBERTa uplift</div><div class='kpi-value'>+5.1%</div><div class='kpi-sub'>over SVM</div></div>
    </div>
    """, unsafe_allow_html=True)

    metrics    = ["Accuracy","Precision","Recall","F1 Score","F1 Macro"]
    mod_colors = ["#B8B5A8","#6B6560","#4A7C59"]

    fig_r = go.Figure()
    for i, row in df_m.iterrows():
        rgb = tuple(int(mod_colors[i].lstrip("#")[j:j+2],16) for j in (0,2,4))
        fig_r.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics], theta=metrics, fill='toself', name=row["Model"],
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)", line=dict(color=mod_colors[i], width=2)
        ))
    fig_r.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0.4,1.0], gridcolor="rgba(184,181,168,0.4)", color="#6B6560", tickformat=".0%"),
            angularaxis=dict(gridcolor="rgba(184,181,168,0.4)", color="#2C2825")),
        title=dict(text="Model Radar", font=dict(family="Lora", size=14, color="#2C2825")),
        **PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig_r, use_container_width=True)

    fig_b = go.Figure()
    for i, m in enumerate(metrics):
        fig_b.add_trace(go.Bar(name=m, x=df_m["Model"], y=df_m[m],
            marker_color=["#4A7C59","#C9973A","#B85450","#B8B5A8","#6B6560"][i],
            marker_line_width=0, text=[f"{v:.1%}" for v in df_m[m]], textposition="outside",
            textfont=dict(size=11, color="#2C2825")))
    fig_b.update_layout(barmode="group",
        title=dict(text="All Metrics", font=dict(family="Lora", size=14, color="#2C2825")),
        xaxis=dict(color="#6B6560"),
        yaxis=dict(gridcolor="rgba(184,181,168,0.3)", color="#6B6560", tickformat=".0%", range=[0,1.08]),
        **PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("""
    <div class='section-card'>
        <div class='section-label'>Key takeaways</div>
        <div class='insight'>RoBERTa leads at 87.6% accuracy — robust pre-training gives superior contextual understanding.</div>
        <div class='insight info'>BERT gains +11.3pp in F1 Macro over SVM, showing better handling of minority classes.</div>
        <div class='insight warn'>Neutral sentiment remains the hardest class across all models.</div>
        <div class='insight'>SVM is a fast, solid baseline — useful when inference speed matters.</div>
    </div>
    """, unsafe_allow_html=True)