import pandas as pd
import numpy as np
import pickle

# Load test data
df_test = pd.read_csv("rl_input_data.csv")

with open("rl_decision_q.pkl", "rb") as f:
    Q = pickle.load(f)

def get_state(row):
    score = row["IDS_Score"]
    score_bin = int(score * 10)
    score_bin = min(score_bin, 10)

    port = int(row["Destination_Port"])
    port_group = port // 1000
    port_group = min(port_group, 10)

    return (score_bin, port_group)

TP = TN = FP = FN = 0

for _, row in df_test.iterrows():
    state = get_state(row)

    if state in Q:
        action = int(np.argmax(Q[state]))  # 0 = ALLOW, 1 = BLOCK
    else:
        action = 1 if row["IDS_Score"] >= 0.8 else 0

    true_label = int(row["True_Label"])

    if action == 1 and true_label == 1:
        TP += 1
    elif action == 1 and true_label == 0:
        FP += 1
    elif action == 0 and true_label == 0:
        TN += 1
    elif action == 0 and true_label == 1:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
fpr = FP / (FP + TN + 1e-6)

print("\n---- RL DECISION LAYER + IDS RESULTS ----")
print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("FPR      :", round(fpr, 4))

# ---- SAVE TO CSV FOR PLOTS ----
df_out = pd.DataFrame([{
    "Accuracy": round(accuracy, 6),
    "Precision": round(precision, 6),
    "Recall": round(recall, 6),
    "FPR": round(fpr, 6)
}])

df_out.to_csv("results/rl_results.csv", index=False)
print("\nSaved → results/rl_results.csv")
