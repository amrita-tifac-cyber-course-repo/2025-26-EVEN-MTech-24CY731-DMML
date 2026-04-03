import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
from dotenv import load_dotenv
import sys

# Standard helper functions from unified_evaluator
def generate_critique(client, prompt, response): 
    sys_prompt = "You are a rigorous factual auditor. Produce a 100% FACTUAL and CORRECTED version of the response. Output ONLY the text."
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Prompt: {prompt}\nResponse: {response}"}]
    completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.0, max_tokens=300)
    return completion.choices[0].message.content.strip()

class NativeSelfCheckNLI:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
    def predict(self, sent, sample):
        inputs = self.tokenizer(sample, sent, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad(): outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return 1.0 - probs[0][2].item() # 1 - Entailment

def run_demo():
    # sys.stdout.reconfigure(encoding='utf-8') # Ensured for Windows terminal
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Config", ".env"))
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    device = torch.device("cpu")
    nli_engine = NativeSelfCheckNLI(device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "datasets", "hallucination_dataset_clean.csv")
    df = pd.read_csv(dataset_path)
    
    # Pick specific samples for maximum clarity in demo
    fact_sample = df.iloc[0] # "What happens if you eat watermelon seeds?"
    hall_sample = df[df['label'] == 1].iloc[0] # Pick the first available hallucination
    
    demo_cases = [("FACTUAL CASE", fact_sample), ("HALLUCINATED CASE", hall_sample)]
    
    THRESHOLD = 0.18

    print("\n" + "="*80)
    print("LIVE DEMONSTRATION")
    print("="*80 + "\n")

    for title, row in demo_cases:
        print(f"\n>>> TESTING {title}")
        print(f"PROMPT   : {row['prompt']}")
        print(f"RESPONSE : {row['response']}\n")
        
        sentences = [s.strip() for s in row['response'].split('.') if len(s.strip()) > 5]
        critique = generate_critique(client, row['prompt'], row['response'])
        print(f"CRITIQUE : {critique[:150]}...\n")
        
        print("SCORING SENTENCES:")
        max_score = 0
        for sent in sentences:
            score = nli_engine.predict(sent, critique)
            max_score = max(max_score, score)
            status = "[HALLUCINATION]" if score >= THRESHOLD else "[TRUTH]"
            # Using clean text to avoid encoding errors
            print(f"  {status} Score: {score:.2f} | {sent}")
        
        print(f"\nFINAL VERDICT: {'HALLUCINATION DETECTED' if max_score >= THRESHOLD else 'FACTUAL OUTPUT'}")
        print("-" * 50)

if __name__ == "__main__":
    run_demo()
