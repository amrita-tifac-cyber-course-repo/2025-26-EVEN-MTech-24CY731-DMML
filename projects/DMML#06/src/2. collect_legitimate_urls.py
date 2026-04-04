"""
2. collect_legitimate_urls.py — OPTIMAL BIAS-FREE VERSION

Key fixes:
1. Mirrors actual PhishTank URL depth distribution (3/67/11/19%)
2. ZERO phishing keywords in any path
3. URL length distribution comparable to phishing URLs
"""

import pandas as pd
import random
from urllib.parse import urlparse

random.seed(42)

# ALL paths verified keyword-free:
# No login/verify/account/bank/payment/secure/update/support/
# checkout/billing/card/credential/password/service/care etc.

PATHS_D1 = [
    "/about", "/about-us", "/contact", "/pricing",
    "/blog", "/news", "/resources", "/documentation",
    "/faq", "/help", "/home", "/features",
    "/products", "/solutions", "/partners",
    "/team", "/mission", "/press", "/investors",
    "/events", "/videos", "/gallery", "/media",
    "/downloads", "/changelog", "/roadmap",
    "/community", "/forum", "/newsletter",
    "/awards", "/testimonials", "/how-it-works",
    "/courses", "/programs", "/admissions",
    "/faculty", "/research", "/library",
    "/shop", "/deals", "/clearance",
    "/entertainment", "/sports", "/health",
    "/technology", "/business", "/politics",
    "/science", "/magazine", "/podcast",
    "/jobs",
]

PATHS_D2 = [
    "/products/electronics", "/products/clothing",
    "/products/home-garden", "/products/books",
    "/products/music", "/products/gaming",
    "/products/furniture", "/products/sports",
    "/category/laptops", "/category/phones",
    "/category/tablets", "/category/cameras",
    "/category/headphones", "/category/watches",
    "/blog/technology", "/blog/tutorials",
    "/blog/announcements", "/blog/tips",
    "/news/technology", "/news/business",
    "/news/world", "/news/sports",
    "/news/entertainment", "/news/science",
    "/about/company", "/about/history",
    "/about/team", "/about/mission",
    "/developer/docs", "/developer/api",
    "/developer/sdk", "/developer/guides",
    "/courses/computer-science", "/courses/mathematics",
    "/courses/physics", "/courses/engineering",
    "/programs/mba", "/programs/phd",
    "/shop/deals", "/shop/new-arrivals",
    "/brand/samsung", "/brand/apple",
    "/brand/sony", "/brand/lg",
    "/sports/football", "/sports/cricket",
    "/entertainment/movies", "/entertainment/music",
    "/health/nutrition", "/health/fitness",
]

PATHS_D3 = [
    "/products/electronics/laptops",
    "/products/electronics/phones",
    "/products/clothing/men/shirts",
    "/products/clothing/women/dresses",
    "/products/home-garden/furniture",
    "/blog/tutorials/python-basics",
    "/blog/tutorials/web-development",
    "/blog/technology/artificial-intelligence",
    "/news/technology/machine-learning",
    "/news/business/startups",
    "/about/company/leadership/team",
    "/developer/docs/api/v1",
    "/developer/docs/api/v2",
    "/developer/guides/getting-started",
    "/courses/computer-science/algorithms",
    "/courses/mathematics/calculus",
    "/category/laptops/gaming-laptops",
    "/category/phones/android",
    "/category/phones/ios",
    "/shop/deals/flash-sale",
    "/brand/samsung/phones",
    "/brand/apple/macbooks",
]

QUERY_PARAMS = [
    "", "", "", "", "", "", "",
    "?lang=en", "?ref=home",
    "?page=1", "?page=2",
    "?category=all", "?sort=popular",
    "?view=list", "?id=12345",
    "?tab=overview", "?filter=new",
]


def make_url(domain, depth):
    scheme = "https://" if random.random() > 0.3 else "http://"
    prefix = "www." if random.random() < 0.15 else ""
    base   = f"{scheme}{prefix}{domain}"
    if depth == 0:
        return base
    elif depth == 1:
        return f"{base}{random.choice(PATHS_D1)}{random.choice(QUERY_PARAMS)}"
    elif depth == 2:
        return f"{base}{random.choice(PATHS_D2)}{random.choice(QUERY_PARAMS)}"
    else:
        return f"{base}{random.choice(PATHS_D3)}{random.choice(QUERY_PARAMS)}"


print("Loading Tranco dataset...")
data    = pd.read_csv("tranco_legit_urls.csv", header=None, names=["rank","domain"])
domains = data["domain"].iloc[10:20010].reset_index(drop=True)
n       = len(domains)

# Match PhishTank path depth distribution exactly
depth_choices = (
    [0] * int(n * 0.032) +
    [1] * int(n * 0.668) +
    [2] * int(n * 0.105) +
    [3] * int(n * 0.195)
)
while len(depth_choices) < n:
    depth_choices.append(1)
depth_choices = depth_choices[:n]
random.shuffle(depth_choices)

print(f"Generating {n} bias-corrected URLs...")
urls = [make_url(d, depth_choices[i]) for i, d in enumerate(domains)]

df = pd.DataFrame({"url": urls, "label": 0})
df.to_csv("final_legit_urls.csv", index=False)

# Verify
KEYWORDS = ["login","signin","verify","account","bank","payment",
            "checkout","secure","update","support","credential",
            "password","invoice","wallet","transfer","suspended"]
contaminated = sum(1 for u in urls if any(k in u.lower() for k in KEYWORDS))

depths = [urlparse(u if u.startswith('http') else 'http://'+u).path.count('/') for u in urls]
print(f"\n✅ Stats:")
print(f"   Total          : {len(urls)}")
print(f"   Keyword-free   : {len(urls)-contaminated}/{len(urls)}")
print(f"   Depth 0 (bare) : {depths.count(0)} ({depths.count(0)/n*100:.1f}%) ← phishing: 3.2%")
print(f"   Depth 1        : {depths.count(1)} ({depths.count(1)/n*100:.1f}%) ← phishing: 66.8%")
print(f"   Depth 2        : {depths.count(2)} ({depths.count(2)/n*100:.1f}%) ← phishing: 10.5%")
d3 = sum(1 for d in depths if d>=3)
print(f"   Depth 3+       : {d3} ({d3/n*100:.1f}%) ← phishing: 19.4%")
print(f"   URL length mean: {sum(len(u) for u in urls)/len(urls):.1f} ← phishing: 48.3")
print(f"\nSample URLs:")
for u in urls[:6]: print(f"   {u}")
print("\nLegitimate URLs collected:", len(df))