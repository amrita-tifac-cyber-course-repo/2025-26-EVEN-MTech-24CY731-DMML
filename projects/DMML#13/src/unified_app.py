import streamlit as st
import os
import time
from groq import Groq
import torch
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="Unified Dashboard", page_icon="", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stTextArea>div>div>textarea { background-color: #262730; color: white; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    .prediction-box { padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("Unified Hallucination Engine")
st.markdown("---")

# Environment & Constants
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Config", ".env"))
api_key = os.environ.get("GROQ_API_KEY")
THRESHOLD = 0.18

# Logic Components
class NativeSelfCheckNLI:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
        
    def predict(self, sent, critique):
        inputs = self.tokenizer(critique, sent, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Probabilities: [Contradiction, Neutral, Entailment]
        return {
            "contradiction": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "entailment": probs[0][2].item(),
            "inconsistency": 1.0 - probs[0][2].item()
        }

@st.cache_resource
def load_nli_model():
    return NativeSelfCheckNLI(device=torch.device("cpu"))

def generate_critique(client, prompt, response):
    sys_prompt = "You are a rigorous factual auditor. Produce a 100% FACTUAL and CORRECTED version of the response. Output ONLY the text."
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Prompt: {prompt}\nResponse: {response}"}]
    completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.0, max_tokens=300)
    return completion.choices[0].message.content.strip()

def get_topic(client, prompt):
    sys_prompt = "Classify this prompt into one category: Science, History, Politics, Technology, Medicine, Culture, Daily Life, Other. Output ONLY the category name."
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    completion = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.0, max_tokens=10)
    return completion.choices[0].message.content.strip()

# UI Layout
if not api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

client = Groq(api_key=api_key)
selfcheck = load_nli_model()

# Sidebar: Sensitivity Controls
st.sidebar.markdown("###  System Calibration")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.40, help="Lower = More Sensitive. Higher = More Conservative.")
st.sidebar.markdown("---")
st.sidebar.write("**Current Strategy:** 1 - Entailment")
st.sidebar.write("**Model:** RoBERTa-Large (Quantized)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Verification")
    with st.form("checker_form"):
        user_prompt = st.text_area("User Prompt", placeholder="e.g., What happens if you eat watermelon seeds?", height=100)
        llm_response = st.text_area("LLM Response", placeholder="e.g., The seeds pass through your system...", height=150)
        run_btn = st.form_submit_button("RUN FACT-CHECK")

with col2:
    st.subheader("Engine Analysis")
    if run_btn:
        if not user_prompt or not llm_response:
            st.warning("Please enter both a prompt and a response to analyze.")
        else:
            with st.status("Analyzing...", expanded=True) as status:
                st.write("Detecting domain/topic...")
                topic = get_topic(client, user_prompt)
                
                st.write("Generating factual critique baseline...")
                critique = generate_critique(client, user_prompt, llm_response)
                
                st.write("Performing NLI sentence cross-reference...")
                sentences = [s.strip() for s in llm_response.split('.') if len(s.strip()) > 5]
                if not sentences: sentences = [llm_response]
                
                results = []
                for s in sentences:
                    res = selfcheck.predict(s, critique)
                    results.append(res)
                
                max_score = max([r["inconsistency"] for r in results])
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # Dashboard Metrics
            m1, m2 = st.columns(2)
            m1.metric("Domain", topic)
            m2.metric("Risk Score", f"{max_score:.2f}", delta=f"{max_score-threshold:.2f}", delta_color="inverse")
            
            # Reference Truth Section
            with st.expander("View Reference Truth", expanded=False):
                st.info(critique)
                st.caption("This baseline is used as the 'Ground Truth' for NLI comparison.")

            st.markdown("### Sentence Breakdown")
            for sent, res in zip(sentences, results):
                score = res["inconsistency"]
                # Triple-Label Logic: Lowered contradiction threshold for more aggressive 'Red' flagging
                if score >= threshold:
                    if res["contradiction"] > 0.25:
                        st.error(f"🔴 **[Score: {score:.2f} - HALLUCINATION]** {sent}")
                    else:
                        st.warning(f"🟡 **[Score: {score:.2f} - UNCERTAIN/UNSUPPORTED]** {sent}")
                else:
                    st.success(f"🟢 **[Score: {score:.2f} - FACTUAL]** {sent}")
            
            if max_score >= threshold:
                is_hallucination = any(r["contradiction"] > 0.25 and r["inconsistency"] >= threshold for r in results)
                st.markdown("---")
                if is_hallucination:
                    st.error("### Final Verdict: HALLUCINATION DETECTED")
                else:
                    st.warning("### Final Verdict: UNCERTAIN / INCONSISTENCY DETECTED")
            else:
                st.markdown("---")
                st.success("### Final Verdict: FACTUAL OUTPUT")
    else:
        st.info("Enter your text in the form and click 'RUN FACT-CHECK' to begin.")
