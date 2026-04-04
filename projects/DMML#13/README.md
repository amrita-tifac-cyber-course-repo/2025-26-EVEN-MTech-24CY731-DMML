# DMML
# Hallucination Detection Engine

## Project Overview
This is a hallucination detection system designed to identify factual inconsistencies in Large Language Model (LLM) outputs. Developed for MiniProject, this engine achieves Accuracy (75%+) by combining stochastic verification with a multi-stage Self-Critique NLI architecture.

---
## Dataset
Link: https://drive.google.com/drive/folders/1WBXphGoEC7KFFZyORsOcQIMB2lVC7gp2?usp=sharing

## Folder Structure & File Descriptions

### Logic/
- `unified_app.py`: The main Streamlit Dashboard. This is the interactive UI where you can input a prompt and response to get a real-time factual audit.
- `unified_evaluator.py`: The Academic Evaluation Suite. This script runs the engine against a balanced dataset (Facts vs. Hallucinations) to generate performance metrics and ROC/Confusion plots.
- `demo.py`: A lightweight demonstration script showing the core NLI detection logic in a simplified format.
- `generate_final_plots.py`: A utility script used to consolidate evaluation data and generate the final visual reports found in the Results folder.

### Results/
- This directory contains the hard evidence of the engine's performance:
    - `.png`: ROC Curves, Confusion Matrices, Boxplots, and Distribution Heatmaps for 50, 100, and 200-sample runs.
    - `unified_academic_results_final.csv`: The raw data containing every prediction made during the final evaluation sweep.

### Visuals/
- `system_architecture_flowchart1.png`: A high-level view of the Multi-Stage Verification Pipeline.
- `preprocessing_flowchart1.png`: Details on how the hallucination dataset was cleaned and balanced.
- `model_arch.jpg`: Technical diagram of the Self-Critique NLI core.
- `Datasets.png`: Combination of all the datasets used.

### Config/
- `requirements.txt`: List of all Python dependencies required to run the engine.
- `.env.example`: A template for your environment variables (API Keys).

---

## Setup & Installation

1. Install Dependencies:
   ```bash
   pip install -r Config/requirements.txt
   ```

2. Configure API Keys:
   - Create a `.env` file in the `Config/` folder.
   - Add your GROQ_API_KEY to the file: `GROQ_API_KEY=your_key_here`

3. Run the Dashboard:
   ```bash
   streamlit run Logic/unified_app.py
   ```

4. Run Evaluation:
   ```bash
   python Logic/unified_evaluator.py
   ```

---

##  Technical Highlights
- Self-Critique Architecture: Instead of simple consistency checking, the engine generates a "Critique Baseline" using Llama-3.1-8B as a factual auditor, then cross-references it using NLI.
- RoBERTa-Large NLI Core: Uses a full-precision RoBERTa model to calculate mathematical contradiction scores.
- Performance Optimization: Implements Dynamic Quantization (INT8) to ensure the engine runs fast even on standard CPU hardware.
- High Precision Focus: Optimised for 72% Precision, ensuring that when the system flags a hallucination, it is highly reliable.

---
> Note: This file represents the optimized, organized submission of the MiniProject in Hallucination Detection.
