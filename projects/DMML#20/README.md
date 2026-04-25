
# CTI-Aware Malicious URL Detection using Machine Learning

### Ensemble Stacking with Adversarial Robustness Evaluation

---

## Project Overview

This project focuses on detecting malicious URLs using a machine learning-based approach enhanced with **Cyber Threat Intelligence (CTI)** features.

The system leverages an **ensemble stacking model** to improve detection accuracy and evaluates its robustness against adversarial attacks.

It is designed to support **real-world cybersecurity workflows**, especially in **OSINT and Cyber Threat Intelligence (CTI)** domains.

---

## Objectives

* Detect malicious URLs with high accuracy
* Integrate **CTI-based features** with traditional lexical features
* Build an **ensemble stacking model** (XGBoost + CatBoost + MLP)
* Evaluate **adversarial robustness** of the model
* Support automation in threat intelligence workflows

---

## Methodology

### 1. Data Collection

* Dataset size: **194,620 URLs**

  * Training: 136,234
  * Testing: 58,386

### 2. Feature Engineering

* **Lexical Features (~15):**

  * URL length
  * Number of dots
  * Presence of special characters
  * Domain-related features

* **CTI Features (~5):**

  * Domain reputation
  * Threat intelligence signals
  * Blacklist indicators

### 3. Models Used

* XGBoost
* CatBoost
* Multi-Layer Perceptron (MLP)

### 4. Ensemble Technique

* **Stacking model** combining predictions from base learners

### 5. Adversarial Testing

* Evaluated model performance under **evasion attacks**
* Measured:

  * Accuracy drop
  * Attack Success Rate (ASR)

---

## Results

| Model         | Accuracy  | Robustness  |
| ------------- | --------- | ----------- |
| XGBoost       | High      | Medium      |
| CatBoost      | High      | High        |
| MLP           | Medium    | Low         |
| Stacked Model | Very High | Medium-High |

✔ Ensemble model achieved **better generalization**
✔ CTI features improved detection capability
✔ Adversarial testing revealed **model resilience limitations**


## Technologies Used

* Python
* Scikit-learn
* XGBoost
* CatBoost
* TensorFlow / Keras
* Pandas, NumPy

---

## Applications

* Cyber Threat Intelligence (CTI)
* Phishing detection systems
* Browser security tools
* OSINT automation

---
## How to Run
python src/phase_1_of_dmml_mini_project.py
python src/phase_2_of_dmml_mini_project.py

## Conclusion

This project demonstrates that combining **machine learning with CTI features** significantly improves malicious URL detection.
However, adversarial testing highlights the need for **robust and secure ML models** in cybersecurity applications.

---
