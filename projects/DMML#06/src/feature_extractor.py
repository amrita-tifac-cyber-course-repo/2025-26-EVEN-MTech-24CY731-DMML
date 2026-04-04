import pandas as pd
import re
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# EXPANDED KEYWORD LIST  (was 7, now 40+)
# ─────────────────────────────────────────────
PHISHING_KEYWORDS = [
    # Credential / account
    "login", "signin", "logon", "sign-in", "log-in",
    "account", "accounts", "myaccount",
    "verify", "verification", "validate", "validation",
    "confirm", "confirmation", "authenticate",
    "password", "passwd", "credential", "credentials",
    # Financial
    "bank", "banking", "payment", "pay", "billing",
    "invoice", "checkout", "wallet", "transfer", "wire",
    "card", "creditcard", "debitcard",
    # Security urgency
    "secure", "security", "alert", "warning", "notice",
    "update", "upgrade", "restore", "recover", "recovery",
    "suspended", "locked", "limited", "access", "unlock",
    # Support bait
    "support", "helpdesk", "helpcenter", "service",
    "customer", "care", "refund",
]

# ─────────────────────────────────────────────
# EXPANDED SUSPICIOUS TLD LIST  (was 6, now 30+)
# ─────────────────────────────────────────────
SUSPICIOUS_TLDS = [
    # Classic free/abused TLDs
    ".tk", ".ml", ".ga", ".cf", ".gq",
    # Low-cost / high-abuse gTLDs
    ".xyz", ".top", ".club", ".online", ".site",
    ".icu", ".buzz", ".vip", ".live", ".pro",
    ".space", ".fun", ".info", ".biz", ".name",
    ".pw", ".cc", ".ws", ".su",
    # Country codes frequently abused
    ".ru", ".cn", ".in", ".br",
]


def keyword_feature(url):
    url_lower = url.lower()
    for k in PHISHING_KEYWORDS:
        if k in url_lower:
            return 1
    return 0


def suspicious_tld(domain):
    domain_lower = domain.lower()
    for tld in SUSPICIOUS_TLDS:
        if domain_lower.endswith(tld):
            return 1
    return 0


def has_brand_plus_tld(domain):
    """Brand name combined with suspicious TLD — strong phishing signal."""
    brands = [
        "google", "paypal", "amazon", "facebook", "apple",
        "microsoft", "instagram", "netflix", "bankofamerica",
        "twitter", "linkedin", "dropbox", "adobe", "yahoo",
        "ebay", "chase", "wellsfargo", "citibank"
    ]
    for brand in brands:
        if brand in domain.lower():
            for tld in SUSPICIOUS_TLDS:
                if domain.lower().endswith(tld):
                    return 1
    return 0


def digit_ratio(url):
    """Ratio of digits to total characters — phishing URLs tend to have more."""
    total = len(url)
    if total == 0:
        return 0
    return round(sum(c.isdigit() for c in url) / total, 4)


def extract_features(url):
    parsed = urlparse("http://" + url)
    domain = parsed.netloc
    path = parsed.path

    features = {}

    # ── Structural ───────────────────────────────────────────
    features["url_length"]      = len(url)
    features["num_dots"]        = url.count(".")
    features["num_hyphens"]     = url.count("-")
    features["num_at"]          = url.count("@")
    features["num_question"]    = url.count("?")
    features["num_percent"]     = url.count("%")
    features["num_equal"]       = url.count("=")
    features["num_digits"]      = sum(c.isdigit() for c in url)
    features["num_letters"]     = sum(c.isalpha() for c in url)
    features["num_directories"] = path.count("/")
    features["use_ip"]          = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    features["num_subdomain"]   = max(domain.count(".") - 1, 0)

    # ── URL shortening ───────────────────────────────────────
    shorteners = [
        "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly",
        "buff.ly", "is.gd", "rb.gy", "tiny.cc", "shorte.st"
    ]
    features["short_url"] = 1 if any(s in url for s in shorteners) else 0

    # ── Content signals ──────────────────────────────────────
    features["keyword_flag"]    = keyword_feature(url)
    features["suspicious_tld"]  = suspicious_tld(domain)

    # ── NEW: additional features ─────────────────────────────
    features["digit_ratio"]         = digit_ratio(url)
    features["brand_tld_combo"]     = has_brand_plus_tld(domain)
    features["has_punycode"]        = 1 if "xn--" in url.lower() else 0
    features["num_underscores"]     = url.count("_")
    features["num_ampersand"]       = url.count("&")
    features["has_redirect"]        = 1 if "redirect" in url.lower() or "url=" in url.lower() else 0
    features["path_length"]         = len(path)
    features["num_fragments"]       = 1 if "#" in url else 0

    return features


# ── Run standalone ────────────────────────────────────────────
if __name__ == "__main__":
    data = pd.read_csv("../1. dataset/final_dataset.csv")
    rows = []
    for _, row in data.iterrows():
        features = extract_features(row["url"])
        features["label"] = row["label"]
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv("url_features.csv", index=False)
    print(f"Feature dataset generated: {len(df)} rows, {len(df.columns)-1} features")
    print("Features:", list(df.columns[:-1]))