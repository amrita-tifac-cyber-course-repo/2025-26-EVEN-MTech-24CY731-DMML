"""
app.py — Zero-Day Phishing URL Detector
Enhanced UI
"""

import os, sys, re, joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "4. adversarial_detection"))
sys.path.insert(0, os.path.join(BASE, "2. feature_extraction"))

from detect_adversarial import (
    detect_adversarial, normalize_homoglyphs, normalize_leet, BRAND_LIST
)
from feature_extractor import extract_features, SUSPICIOUS_TLDS

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard — Zero-Day Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 4rem 4rem 4rem; max-width: 1100px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .8rem;
    line-height: 1.15;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* ── Search box ── */
.search-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

/* ── Override streamlit input ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    height: auto !important;
    transition: border .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #475569 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    transition: all .2s !important;
    width: 100%;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    transition: all .15s !important;
    width: 100%;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
}

/* ── Result cards ── */
.result-phishing {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    animation: fadeInUp .4s ease;
}
.result-legit {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.06));
    border: 1.5px solid rgba(34,197,94,0.4);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    animation: fadeInUp .4s ease;
}
.result-icon { font-size: 2.5rem; margin-bottom: .5rem; }
.result-title-phish {
    font-size: 1.6rem; font-weight: 700; color: #f87171; margin: 0 0 .3rem;
}
.result-title-legit {
    font-size: 1.6rem; font-weight: 700; color: #4ade80; margin: 0 0 .3rem;
}
.result-layer {
    font-size: 0.82rem; color: #94a3b8;
    background: rgba(255,255,255,0.06);
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; margin-bottom: 1rem;
}
.reason-item {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 6px 0;
    color: #fca5a5;
    font-size: 0.88rem;
}

/* ── Probability bar ── */
.prob-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.prob-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    min-width: 120px;
}
.prob-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing:.08em; }
.prob-value-phish { font-size: 1.6rem; font-weight: 700; color: #f87171; }
.prob-value-legit { font-size: 1.6rem; font-weight: 700; color: #4ade80; }

/* ── Layer pills ── */
.layers-row {
    display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.5rem;
}
.layer-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 10px;
    font-size: 0.82rem; font-weight: 500;
}
.l1 { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.35); color: #a5b4fc; }
.l2 { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #fcd34d; }
.l3 { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.3);  color: #86efac; }

/* ── Quick test chips ── */
.chip-label {
    font-size: 0.75rem; color: #475569;
    text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: .5rem;
}

/* ── Feature table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stDataFrame thead tr th {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important; font-size: 0.8rem;
}

/* ── Batch metrics ── */
.metric-row {
    display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center; min-width: 100px;
}
.metric-card-val { font-size: 2rem; font-weight: 700; color: #fff; }
.metric-card-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; color: #64748b;
    font-size: 0.88rem; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
}

/* ── Animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── How it works cards ── */
.how-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.how-card h4 { color: #e2e8f0; margin: 0 0 .5rem; font-size: 1rem; }
.how-card p  { color: #64748b; font-size: 0.85rem; margin: 0; line-height: 1.6; }

/* ── textarea ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "3. model_training", "phishing_model.pkl"))

model = load_model()

# ── Helpers ────────────────────────────────────────────────────────
def brand_tld_attack(domain):
    phishing_words = ["login", "verify", "secure", "account", "update", "signin"]
    domain_l = domain.lower()
    normalized = normalize_leet(normalize_homoglyphs(domain_l))
    normalized = re.sub(r"[-_.]", "", normalized)
    for brand in BRAND_LIST:
        if brand in normalized:
            for tld in SUSPICIOUS_TLDS:
                if domain_l.endswith(tld):
                    return True, f"Brand '{brand}' combined with suspicious TLD '{tld}'"
            for word in phishing_words:
                if word in normalized:
                    return True, f"Brand '{brand}' combined with phishing keyword '{word}'"
    return False, ""

def detect(url):
    from urllib.parse import urlparse
    url = url.strip()
    parsed = urlparse(url if url.startswith(("http://","https://")) else "http://"+url)
    domain = parsed.netloc

    suspicious, reasons = detect_adversarial(url)
    if suspicious:
        return {"verdict":"PHISHING","layer":"Layer 1 — Adversarial / Zero-Day","reasons":reasons,
                "proba_phishing":1.0,"proba_legit":0.0,"features":None}

    is_brand, brand_msg = brand_tld_attack(domain)
    if is_brand:
        return {"verdict":"PHISHING","layer":"Layer 2 — Brand Impersonation","reasons":[brand_msg],
                "proba_phishing":1.0,"proba_legit":0.0,"features":None}

    clean_url = url.replace("http://","").replace("https://","")
    features  = extract_features(clean_url)
    df        = pd.DataFrame([features])
    prediction= model.predict(df)[0]
    proba     = model.predict_proba(df)[0]
    return {"verdict":"PHISHING" if prediction==1 else "LEGITIMATE",
            "layer":"Layer 3 — Machine Learning (Random Forest)",
            "reasons":[],"proba_phishing":float(proba[1]),
            "proba_legit":float(proba[0]),"features":features}

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ Zero-Day Phishing Detection</div>
    <h1>PhishGuard</h1>
    <p>Detect phishing URLs instantly using a three-layer hybrid system —
    adversarial pattern rules, brand impersonation detection, and machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><div class="stat-val">96.65%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-item"><div class="stat-val">40,000</div><div class="stat-lbl">Training URLs</div></div>
    <div class="stat-item"><div class="stat-val">3 Layers</div><div class="stat-lbl">Detection</div></div>
    <div class="stat-item"><div class="stat-val">23</div><div class="stat-lbl">Features</div></div>
    <div class="stat-item"><div class="stat-val">0.9943</div><div class="stat-lbl">AUC-ROC</div></div>
</div>
""", unsafe_allow_html=True)

# Layer pills
st.markdown("""
<div class="layers-row">
    <div class="layer-pill l1">⚡ Layer 1 — Adversarial Rules</div>
    <div class="layer-pill l2">🔍 Layer 2 — Brand Impersonation</div>
    <div class="layer-pill l3">🤖 Layer 3 — Random Forest ML</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Single URL", "📋  Batch Scan", "📖  How It Works"])

# ─── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialise session state
    if "url_val" not in st.session_state:
        st.session_state["url_val"] = ""

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url_input = st.text_input(
            "URL", value=st.session_state["url_val"],
            placeholder="🔗  Enter a URL — e.g. g00gle.com or paypal-login.xyz",
            key="url_val", label_visibility="collapsed"
        )
    with col_btn:
        check = st.button("Scan →", type="primary", use_container_width=True)

    # Quick test chips
    st.markdown('<div class="chip-label">Quick test</div>', unsafe_allow_html=True)
    cols = st.columns(7)
    examples = ["g00gle.com","paypa1-login.xyz","gooooogle.com",
                "paypal.com.secure-site.tk","google.com","amazon.com","microsoft.com"]
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"chip_{ex}"):
                st.session_state["url_val"] = ex
                st.rerun()

    # Result
    if check and url_input:
        with st.spinner(""):
            result = detect(url_input)

        if result["verdict"] == "PHISHING":
            st.markdown(f"""
            <div class="result-phishing">
                <div class="result-icon">⚠️</div>
                <div class="result-title-phish">Phishing Detected</div>
                <div class="result-layer">{result['layer']}</div>
                {"".join(f'<div class="reason-item">• {r}</div>' for r in result["reasons"])}
                <div class="prob-row">
                    <div class="prob-box">
                        <div class="prob-label">Phishing probability</div>
                        <div class="prob-value-phish">{result['proba_phishing']:.1%}</div>
                    </div>
                    <div class="prob-box">
                        <div class="prob-label">Legitimate probability</div>
                        <div class="prob-value-legit">{result['proba_legit']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-icon">✅</div>
                <div class="result-title-legit">URL is Legitimate</div>
                <div class="result-layer">{result['layer']}</div>
                <div class="prob-row">
                    <div class="prob-box">
                        <div class="prob-label">Legitimate probability</div>
                        <div class="prob-value-legit">{result['proba_legit']:.1%}</div>
                    </div>
                    <div class="prob-box">
                        <div class="prob-label">Phishing probability</div>
                        <div class="prob-value-phish">{result['proba_phishing']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if result["features"]:
            with st.expander("📊 Feature analysis (ML layer)"):
                feat_df = pd.DataFrame([result["features"]]).T.reset_index()
                feat_df.columns = ["Feature", "Value"]
                feat_df["Status"] = feat_df["Value"].apply(
                    lambda v: "🔴 High risk" if v == 1 else ("🟡 Notable" if v > 0.5 else "🟢 Normal")
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ─── TAB 2 ────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    batch_input = st.text_area(
        "Enter one URL per line:",
        height=200,
        placeholder="google.com\npaypa1-login.xyz\ng00gle.com\namazon.com\nlogin-paypal.xyz"
    )

    if st.button("Scan All URLs", type="primary"):
        urls = [u.strip() for u in batch_input.strip().splitlines() if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            results = []
            bar = st.progress(0, text="Scanning...")
            for i, url in enumerate(urls):
                r = detect(url)
                results.append({
                    "URL": url,
                    "Verdict": r["verdict"],
                    "Layer": r["layer"].split("—")[-1].strip(),
                    "Phishing %": f"{r['proba_phishing']:.1%}",
                    "Reason": "; ".join(r["reasons"]) if r["reasons"] else "ML prediction"
                })
                bar.progress((i+1)/len(urls), text=f"Scanning {i+1}/{len(urls)}...")
            bar.empty()

            df_r = pd.DataFrame(results)
            ph = (df_r["Verdict"]=="PHISHING").sum()
            lg = len(df_r) - ph

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-card-val">{len(df_r)}</div>
                    <div class="metric-card-lbl">Total URLs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#f87171">{ph}</div>
                    <div class="metric-card-lbl">⚠️ Phishing</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#4ade80">{lg}</div>
                    <div class="metric-card-lbl">✅ Legitimate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val">{ph/len(df_r):.0%}</div>
                    <div class="metric-card-lbl">Threat rate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            def color_verdict(val):
                if val == "PHISHING":
                    return "color: #f87171; font-weight: 600"
                return "color: #4ade80; font-weight: 600"

            styled = df_r.style.map(color_verdict, subset=["Verdict"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.download_button("⬇️ Download CSV",
                data=df_r.to_csv(index=False),
                file_name="phishguard_scan.csv", mime="text/csv")

# ─── TAB 3 ────────────────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="how-card">
            <h4>⚡ Layer 1 — Adversarial / Zero-Day</h4>
            <p>Rule-based detection that catches attacks no ML model can — typosquatting
            (Levenshtein distance ≤ 2), Unicode homoglyphs (Cyrillic/Greek lookalikes),
            leet-speak substitutions (0→o, 1→l, 3→e), repeated character patterns,
            domain shadowing, and IDN punycode attacks.</p>
        </div>
        <div class="how-card">
            <h4>🔍 Layer 2 — Brand Impersonation</h4>
            <p>Detects when a known brand name (Google, PayPal, Amazon, Apple, Microsoft etc.)
            is combined with a suspicious TLD (.xyz, .tk, .ml, .top...) or a phishing keyword
            (login, verify, secure, account, update).</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="how-card">
            <h4>🤖 Layer 3 — Machine Learning</h4>
            <p>Random Forest classifier trained on 40,000 URLs (20k phishing from PhishTank,
            20k legitimate from Tranco Top 1M). Uses 23 handcrafted URL features. 
            Accuracy: 96.65% | Precision: 97.19% | Recall: 96.07% | F1: 96.63% | AUC-ROC: 0.9943 | 5-Fold CV: 0.9669 ± 0.0012.</p>
        </div>
        <div class="how-card">
            <h4>📊 Feature Engineering (23 features)</h4>
            <p>URL length, dot/hyphen/digit counts, digit ratio, path depth, subdomain count,
            IP in URL, URL shortener detection, phishing keywords (40+), suspicious TLDs (30+),
            brand+TLD combo, punycode, redirect patterns, fragment count.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-card">
        <h4>🎯 Key Novelty vs Related Work</h4>
        <p>Unlike WebPhish (CNN on URL+HTML), Boolean Algebra detector, or RNN-LSTM models,
        PhishGuard adds a dedicated zero-day adversarial module that catches Unicode homoglyph attacks
        (аpple.com using Cyrillic а), leet-speak substitutions (paypa1.com), typosquatting within
        edit-distance 2, IDN punycode abuse, and domain shadowing — all without needing training data
        for these patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Model | Accuracy | AUC | Zero-Day |
    |-------|----------|-----|----------|
    | **PhishGuard (ours)** | **96.65%** | **0.9943** | **✅ Yes** |
    | WebPhish (2024) | 98.10% | — | ❌ No |
    | Boolean Algebra (2026) | 89.10% | 0.91 | ❌ No |
    | RNN-LSTM (2026) | ~97.00% | — | ❌ No |
    """)