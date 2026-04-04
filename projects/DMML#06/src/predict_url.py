"""
predict_url.py — Zero-Day Phishing URL Detector (CLI)

Run from Project Codes folder:
    python "5. prediction/predict_url.py"

Batch mode:
    python "5. prediction/predict_url.py" test_urls.txt
"""

import sys, os, re, joblib
import pandas as pd
from urllib.parse import urlparse

# ── Path setup ─────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "4. adversarial_detection"))
sys.path.insert(0, os.path.join(BASE, "2. feature_extraction"))

from detect_adversarial import (
    detect_adversarial, normalize_homoglyphs, normalize_leet,
    BRAND_LIST
)
from feature_extractor import (
    extract_features, SUSPICIOUS_TLDS, PHISHING_KEYWORDS
)

# ── Load model ─────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE, "3. model_training", "phishing_model.pkl")
model = joblib.load(MODEL_PATH)


def brand_tld_attack(domain):
    phishing_words = ["login", "verify", "secure", "account", "update", "signin"]
    domain_l = domain.lower()
    normalized = normalize_leet(normalize_homoglyphs(domain_l))
    normalized = re.sub(r"[-_.]", "", normalized)
    for brand in BRAND_LIST:
        if brand in normalized:
            for tld in SUSPICIOUS_TLDS:
                if domain_l.endswith(tld):
                    return True, f"Brand impersonation: '{brand}' + suspicious TLD '{tld}'"
            for word in phishing_words:
                if word in normalized:
                    return True, f"Brand impersonation: '{brand}' + phishing keyword '{word}'"
    return False, ""


def predict_single(url):
    url = url.strip()
    parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
    domain = parsed.netloc
    path   = parsed.path

    print(f"\n{'─'*55}")
    print(f"  URL   : {url}")
    print(f"  Domain: {domain}")
    print(f"{'─'*55}")

    # Layer 1 — adversarial
    suspicious, reasons = detect_adversarial(url)
    if suspicious:
        print("  LAYER 1 — Adversarial Pattern Detected")
        for r in reasons:
            print(f"    • {r}")
        print("\n  ⚠  VERDICT: PHISHING (Adversarial / Zero-Day)")
        return "phishing", "adversarial"

    # Layer 2 — brand impersonation
    is_brand, brand_reason = brand_tld_attack(domain)
    if is_brand:
        print(f"  LAYER 2 — {brand_reason}")
        print("\n  ⚠  VERDICT: PHISHING (Brand Impersonation)")
        return "phishing", "brand_impersonation"

    # Layer 3 — ML
    clean_url = url.replace("http://", "").replace("https://", "")
    features  = extract_features(clean_url)
    df        = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    proba      = model.predict_proba(df)[0]

    confidence = proba[1] if prediction == 1 else proba[0]
    label      = "PHISHING" if prediction == 1 else "LEGITIMATE"
    emoji      = "⚠ " if prediction == 1 else "✅"

    print(f"  LAYER 3 — ML Prediction (Random Forest)")
    print(f"    Phishing probability  : {proba[1]:.3f}")
    print(f"    Legitimate probability: {proba[0]:.3f}")
    print(f"\n  {emoji} VERDICT: {label}  (confidence: {confidence:.1%})")
    return label.lower(), "ml"


def predict_batch(filepath):
    with open(filepath) as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"\nBatch mode — {len(urls)} URLs from '{filepath}'")
    results = []
    for url in urls:
        verdict, layer = predict_single(url)
        results.append({"url": url, "verdict": verdict, "detected_by": layer})
    df = pd.DataFrame(results)
    print(f"\n{'='*55}\nBATCH SUMMARY\n{'='*55}")
    print(df.to_string(index=False))
    phishing_count = (df["verdict"] == "phishing").sum()
    print(f"\nPhishing: {phishing_count}/{len(df)}  |  Legitimate: {len(df)-phishing_count}/{len(df)}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        predict_batch(sys.argv[1])
    else:
        print("\nZero-Day Phishing URL Detector")
        print("Type a URL to check, or 'quit' to exit.\n")
        while True:
            try:
                url = input("Enter URL: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if url.lower() in ("quit", "exit", "q"):
                break
            if url:
                predict_single(url)