import pandas as pd

# Load PhishTank dataset
data = pd.read_csv("phishtank_phishing_urls.csv")

# Extract URLs
phishing_urls = data["url"][:20000]

df = pd.DataFrame(phishing_urls, columns=["url"])
df["label"] = 1   # 1 = Phishing

df.to_csv("final_phishing_urls.csv", index=False)

print("Phishing URLs collected:", len(df))