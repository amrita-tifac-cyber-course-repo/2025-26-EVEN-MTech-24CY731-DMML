import pandas as pd

data = pd.read_csv("merged_dataset.csv")

def clean_url(url):
    url = url.replace("http://", "")
    url = url.replace("https://", "")
    return url

data["url"] = data["url"].apply(clean_url)

data.to_csv("cleaned_dataset.csv", index=False)

print("Dataset cleaned successfully")