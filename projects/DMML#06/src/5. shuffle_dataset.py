import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save final dataset
df.to_csv("final_dataset.csv", index=False)

print("Dataset shuffled successfully")
print("Final dataset size:", df.shape)