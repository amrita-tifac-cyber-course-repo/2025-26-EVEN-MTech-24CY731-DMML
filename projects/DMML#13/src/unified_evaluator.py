import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys
import re

try:
    from groq import Groq
except ImportError:
    os.system(f"{sys.executable} -m pip install groq scikit-learn matplotlib seaborn pandas numpy python-dotenv transformers")
    from groq import Groq

from dotenv import load_dotenv

# --- IDEA 1 COMPONENTS (NLI CORE) ---
def generate_critique(client, prompt, response): 
    sys_prompt = "You are a rigorous factual auditor. Your task is to identify and fix hallucinations in the provided response. If the response contains any false claims, fabrications, or minor inaccuracies, produce a 100% FACTUAL and CORRECTED version. Do NOT use buzzwords or meta-talk. If the response is already perfect, repeat it exactly. Be extremely sensitive to subtle lies."
    user_content = f"Prompt: {prompt}\nResponse: {response}"
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0, # Complete deterministic accuracy for auditors
            max_tokens=300
        )
        return [completion.choices[0].message.content.strip()]
    except Exception as e:
        print(f"    [ERR] Critique Error: {re.sub(r'[^a-zA-Z0-9 ]', '', str(e))[:100]}")
        return [""]

class NativeSelfCheckNLI:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)
        
        # --- PERFORMANCE UPGRADE: DYNAMIC QUANTIZATION ---
        self.model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()
        
    def predict(self, sentences, sampled_passages):
        sent_scores = []
        for sent in sentences:
            sample_scores = []
            for sample in sampled_passages:
                if not sample.strip(): continue
                inputs = self.tokenizer(sample, sent, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # index 0: contradiction, index 1: neutral, index 2: entailment
                # We use (1 - entailment_prob) for maximum sensitivity to unsupported claims
                entailment_prob = probs[0][2].item()
                non_entailment_score = 1.0 - entailment_prob
                sample_scores.append(non_entailment_score) 
            if sample_scores:
                sent_scores.append(np.max(sample_scores))
            else:
                sent_scores.append(0.0)
        return sent_scores

# --- IDEA 2 COMPONENTS (TOPIC ROUTING) ---
def get_topic(client, prompt):
    sys_prompt = "You are a classifier. Categorize the given prompt into exactly one of these categories: Science, History, Politics, Technology, Medicine, Culture, Daily Life, Other. Output ONLY the category name."
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", temperature=0.0, max_tokens=10
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "Other"

def main():
    # sys.stdout.reconfigure(encoding='utf-8', line_buffering=True) # Removed for background stability
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Config", ".env"))
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env")
        return
    client = Groq(api_key=api_key)
    
    print("Downloading/Initializing RoBERTa NLI Core Engine...")
    device = torch.device("cpu")
    selfcheck = NativeSelfCheckNLI(device=device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "datasets", "hallucination_dataset_clean.csv")
    print(f"Loading dataset from: {dataset_path}")
    df_full = pd.read_csv(dataset_path)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "final idea", "Results")

    # --- MULTI-SCALE EVALUATOR CONFIG ---
    run_configs = [
        {"n_total": 50, "n_each": 25, "suffix": "50"},
        {"n_total": 100, "n_each": 50, "suffix": "100"},
        {"n_total": 200, "n_each": 100, "suffix": "200"}
    ]
    
    THRESHOLD = 0.18 # Calibrated threshold for 1-Entailment detector (high sensitivity)

    for run in run_configs:
        num_each = run["n_each"]
        suffix = run["suffix"]
        
        print(f"\n\n{'#'*80}")
        print(f" RUNNING EVALUATION FOR {run['n_total']} SAMPLES ({num_each} Factual / {num_each} Hallucinated) ")
        print(f"{'#'*80}\n")
        
        # Balanced sampling
        df_facts = df_full[(df_full['label'] == 0) & (df_full['prompt'].str.len() < 250)].sample(num_each, random_state=42)
        df_halls = df_full[(df_full['label'] == 1) & (df_full['prompt'].str.len() < 250)].sample(num_each, random_state=42)
        df_subset = pd.concat([df_facts, df_halls]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        y_true, y_pred_nli, topics, nli_float_scores = [], [], [], []
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            prompt = row['prompt']
            response = row['response']
            true_label = int(row['label']) 
            
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 5]
            if not sentences: continue
                
            print(f"[{suffix} RUN] Processing {idx+1}/{len(df_subset)}...", end="\r", flush=True)
            
            topic = get_topic(client, prompt)
            critiques = generate_critique(client, prompt, response) 
            sent_scores = selfcheck.predict(sentences, critiques)
            
            max_score = np.max(sent_scores) if sent_scores else 0.0
            pred_label = 1 if max_score >= THRESHOLD else 0
            
            y_true.append(true_label)
            y_pred_nli.append(pred_label)
            topics.append(topic)
            nli_float_scores.append(max_score)
 
        # --- SAVE RESULTS & VISUALS PER RUN ---
        df_results = pd.DataFrame({
            'Topic': topics, 'NLI_Score': nli_float_scores, 
            'Prediction': y_pred_nli, 'TrueLabel': y_true
        })
        df_results.to_csv(os.path.join(output_dir, f'academic_results_{suffix}.csv'), index=False)

        # 1. ROC Curve
        plt.figure(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_true, nli_float_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - Dataset Size {suffix}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'roc_curve_{suffix}.png'))
        
        # 2. Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred_nli)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fact', 'Hallu'], yticklabels=['Fact', 'Hallu'])
        plt.title(f"Confusion Matrix (Size {suffix})")
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{suffix}.png'))
        plt.close('all')

        # --- PERFORMANCE REPORT ---
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_nli, average='binary')
        acc = accuracy_score(y_true, y_pred_nli)
        
        print(f"\n{'='*50}")
        print(f" REPORT FOR {suffix} SAMPLES (THRESHOLD: {THRESHOLD})")
        print(f"{'='*50}")
        print(f" ACCURACY  : {acc:.2%}")
        print(f" PRECISION : {p:.2%}")
        print(f" RECALL    : {r:.2%}")
        print(f" F1-SCORE  : {f1:.2%}")
        print(f" ROC-AUC   : {roc_auc:.4f}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
