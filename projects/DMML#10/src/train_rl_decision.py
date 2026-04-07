import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

# Load RF output data (test split with IDS_Score + True_Label)
df = pd.read_csv("rl_input_data.csv")

# ---- BALANCED TRAINING SAMPLE FOR RL ----
normal = df[df["True_Label"] == 0].sample(50000, random_state=42)
attack = df[df["True_Label"] == 1].sample(50000, random_state=42)

train_df = pd.concat([normal, attack]).sample(frac=1, random_state=42).reset_index(drop=True)

print("RL training sample size:", len(train_df))

# ---- DISCRETIZATION: MAP ROW -> STATE ----
def get_state(row):
    # Bucket IDS_Score into 0..10
    score = row["IDS_Score"]
    score_bin = int(score * 10)
    if score_bin > 10:
        score_bin = 10

    # Coarse port group: 0..10
    port = int(row["Destination_Port"])
    port_group = port // 1000
    if port_group > 10:
        port_group = 10

    return (score_bin, port_group)

# ---- Q-LEARNING SETUP (CONTEXTUAL BANDIT STYLE) ----
# Actions: 0 = ALLOW, 1 = BLOCK
Q = defaultdict(lambda: np.zeros(2))

alpha = 0.1    # learning rate
gamma = 0.0    # no future reward (one-step decision problem)
epsilon = 1.0  # start with full exploration

EPOCHS = 10    # passes over the training data

for epoch in range(EPOCHS):
    # Shuffle each epoch
    train_df = train_df.sample(frac=1, random_state=epoch).reset_index(drop=True)
    total_reward = 0

    for _, row in train_df.iterrows():
        state = get_state(row)
        q_values = Q[state]

        # epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(2)
        else:
            action = int(np.argmax(q_values))

        true_label = int(row["True_Label"])

        # ---- REWARD DESIGN (OPTIMIZATION GOAL) ----
        # Want high recall but NOT insane FPR
        if action == 1 and true_label == 1:      # BLOCK & attack (TP)
            reward = 3
        elif action == 1 and true_label == 0:    # BLOCK & normal (FP)
            reward = -4
        elif action == 0 and true_label == 0:    # ALLOW & normal (TN)
            reward = 3
        elif action == 0 and true_label == 1:    # ALLOW & attack (FN)
            reward = -6
        else:
            reward = -2

        # Q-learning update (no next state, gamma = 0)
        Q[state][action] += alpha * (reward - Q[state][action])
        total_reward += reward

    # epsilon decay (less random over time)
    epsilon = max(0.05, epsilon * 0.9)

    print(f"Epoch {epoch+1}/{EPOCHS} | Total Reward: {total_reward} | Epsilon: {epsilon:.3f}")

# ---- SAVE TRAINED Q-POLICY ----
Q_dict = {state: Q[state] for state in Q}  # convert defaultdict to normal dict

with open("rl_decision_q.pkl", "wb") as f:
    pickle.dump(Q_dict, f)

print("✅ RL decision policy training completed.")
print("✅ Q-table saved to rl_decision_q.pkl")
