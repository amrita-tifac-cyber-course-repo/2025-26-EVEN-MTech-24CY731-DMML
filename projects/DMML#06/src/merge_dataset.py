import pandas as pd

legit = pd.read_csv("final_legit_urls.csv")
phish = pd.read_csv("final_phishing_urls.csv")

dataset = pd.concat([legit, phish])

dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset.to_csv("merged_dataset.csv", index=False)

print("Total dataset size:", len(dataset))
