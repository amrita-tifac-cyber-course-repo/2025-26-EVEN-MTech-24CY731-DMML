import re
import unicodedata
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# BRAND LIST
# ─────────────────────────────────────────────
BRAND_LIST = [
    "google", "paypal", "amazon", "facebook", "apple",
    "microsoft", "instagram", "netflix", "bankofamerica",
    "twitter", "linkedin", "dropbox", "adobe", "yahoo",
    "ebay", "chase", "wellsfargo", "citibank", "barclays",
    "hsbc", "steam", "discord", "whatsapp", "snapchat"
]

# ─────────────────────────────────────────────
# LEET-SPEAK / VISUAL SUBSTITUTIONS
# ─────────────────────────────────────────────
SUBSTITUTIONS = {
    "0": "o", "1": "l", "3": "e", "5": "s",
    "@": "a", "$": "s", "4": "a", "7": "t",
    "!": "i", "|": "l", "8": "b"
}

# ─────────────────────────────────────────────
# UNICODE HOMOGLYPH MAP
# ─────────────────────────────────────────────
HOMOGLYPH_MAP = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o",
    "\u0440": "r", "\u0441": "c", "\u0445": "x",
    "\u0456": "i", "\u0454": "e", "\u0443": "y",
    "\u0432": "b", "\u03b1": "a", "\u03b5": "e",
    "\u03bf": "o", "\u03c1": "p", "\u03bd": "v",
    "\u03c5": "u", "\uff41": "a", "\uff42": "b",
    "\uff43": "c", "\uff44": "d", "\uff45": "e",
    "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4f": "o", "\uff50": "p",
    "\uff53": "s", "\uff54": "t", "\uff55": "u",
    "\uff56": "v", "\uff57": "w", "\uff59": "y",
    "\u0131": "i", "\u00e9": "e", "\u00e0": "a",
    "\u00f8": "o",
}


def extract_domain(url):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    parts = parsed.netloc.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def normalize_homoglyphs(text):
    text = unicodedata.normalize("NFKC", text)
    result = ""
    for ch in text:
        result += HOMOGLYPH_MAP.get(ch, ch)
    return result


def has_mixed_scripts(text):
    scripts = set()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                scripts.add("LATIN")
            elif "CYRILLIC" in name:
                scripts.add("CYRILLIC")
            elif "GREEK" in name:
                scripts.add("GREEK")
    return len(scripts) > 1


def normalize_leet(text):
    for char, letter in SUBSTITUTIONS.items():
        text = text.replace(char, letter)
    return text


def detect_adversarial(url):
    suspicious = False
    reasons = []

    domain = extract_domain(url)
    domain_lower = domain.lower()

    if domain_lower in BRAND_LIST:
        return False, []

    # 1. Unicode homoglyph attack
    normalized_unicode = normalize_homoglyphs(domain_lower)
    if normalized_unicode != domain_lower:
        for brand in BRAND_LIST:
            if brand in normalized_unicode:
                suspicious = True
                reasons.append(
                    f"Unicode homoglyph attack: domain resolves to '{brand}' after normalization "
                    "(Cyrillic/Greek lookalike characters used)"
                )
                break

    # 2. Mixed script (IDN homograph)
    if has_mixed_scripts(domain):
        suspicious = True
        reasons.append(
            "Mixed Unicode scripts in domain (IDN homograph attack — Latin + Cyrillic/Greek mixed)"
        )

    # 3. Levenshtein typosquatting
    try:
        import Levenshtein
        for brand in BRAND_LIST:
            distance = Levenshtein.distance(domain_lower, brand)
            if 0 < distance <= 2:
                suspicious = True
                reasons.append(
                    f"Typosquatting: '{domain_lower}' is {distance} edit(s) away from brand '{brand}'"
                )
    except ImportError:
        pass

    # 4. Leet-speak / character substitution
    # Only fires when actual digit/symbol substitution happened (0→o, 1→l, etc.)
    # Pure brand+keyword combos (paypal-login.xyz) are intentionally left for Layer 2
    leet_normalized = normalize_leet(domain_lower)
    actual_substitution = (leet_normalized != domain_lower)
    stripped = re.sub(r"[-_]", "", leet_normalized)
    stripped = re.sub(r"(login|secure|verify|account|update|signin|support)", "", stripped)
    for brand in BRAND_LIST:
        if brand in stripped and domain_lower not in BRAND_LIST and actual_substitution:
            suspicious = True
            reasons.append(f"Character substitution attack on '{brand}' (e.g. 0→o, 1→l, 3→e)")
            break

    # 5. Repeated characters
    if re.search(r"(.)\1{3,}", domain_lower):
        suspicious = True
        reasons.append("Repeated characters in domain (e.g. gooooogle.com, faceboook.com)")

    # 6. Excessive subdomains
    full = url if url.startswith(("http://", "https://")) else "http://" + url
    dot_count = urlparse(full).netloc.count(".")
    if dot_count >= 3:
        suspicious = True
        reasons.append(f"Excessive subdomains ({dot_count} dots) — possible domain shadowing")

    # 7. Punycode / IDN encoded domain
    if "xn--" in domain_lower:
        suspicious = True
        reasons.append("Punycode encoded domain (xn--) — used in IDN homograph phishing")

    return suspicious, reasons