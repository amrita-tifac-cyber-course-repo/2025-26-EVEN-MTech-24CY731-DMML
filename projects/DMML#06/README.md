# 🔐 Zero-Day Phishing URL Detection Using Machine Learning and Adversarial URL Analysis
> AI & ML/DL Applications in Cybersecurity – DMML Mini Project

## Project Title

**Zero-Day Phishing URL Detection Using Machine Learning and Adversarial URL Analysis**

---

## Team Details

**Team Number:** DMML#06

**Members:**

* CB.SC.P2CYS25007 – Balina Sri Vaishnavi
* CB.SC.P2CYS25006 – Atmala Sai Chandra Koushik

---

## Problem Statement

Phishing attacks are one of the most common cybersecurity threats where attackers create malicious URLs that imitate legitimate websites to steal sensitive information such as login credentials, banking details, and personal data. Traditional phishing detection techniques rely on blacklist databases and signature-based detection methods, which are ineffective in detecting zero-day phishing attacks because newly generated phishing URLs are not present in blacklist databases.

This project aims to detect zero-day phishing URLs using adversarial URL analysis and machine learning techniques by analyzing URL lexical and structural features.

---

## Objectives

The main objectives of this project are:

* Detect zero-day phishing URLs
* Identify adversarial phishing techniques such as typosquatting and homoglyph attacks
* Extract lexical and structural URL features
* Train machine learning models for phishing detection
* Build a multi-layer phishing detection system
* Develop a Streamlit web application for URL prediction

---

## Dataset Details

**Dataset Name:** Phishing and Legitimate URL Dataset

**Dataset Sources:**

* PhishTank – https://phishtank.org/developer_info.php
* Tranco Top 1M – https://tranco-list.eu

**Features Description:**

* URL Length
* Number of Dots
* Number of Hyphens
* Digit Count
* Subdomain Count
* Suspicious TLD Detection
* Presence of IP Address
* Punycode Detection
* Phishing Keywords
* Special Characters Count
* Directory Depth

**Dataset Size:** Approximately 40,000 URLs

---

## Methodology

**Techniques Used:**

* Logistic Regression
* Random Forest

**Workflow:**

1. Collect phishing URLs from PhishTank
2. Collect legitimate URLs from Tranco
3. Merge datasets
4. Clean dataset
5. Shuffle dataset
6. Extract features from URLs
7. Train machine learning model
8. Detect adversarial URLs
9. Predict phishing URLs
10. Streamlit web application

### System Architecture

![System Architecture](images/system_architecture.png)

### Workflow Diagram

![Workflow](images/workflow.png)

### Model Evaluation

#### Model Performance Metrics

| Metric | Logistic Regression | Random Forest (Final Model) |
|-------|---------------------|------------------------------|
| Accuracy | 87.81% | 96.65% |
| Precision | 87.95% | 97.19% |
| Recall | 87.62% | 96.07% |
| F1-Score | 87.79% | 96.63% |
| AUC-ROC | 0.9444 | 0.9943 |
| 5-Fold CV Accuracy | 0.8755 ± 0.0026 | 0.9669 ± 0.0012 |
| False Positive Rate | ~12% | ~2.8% |

#### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

#### ROC Curve

![ROC Curve](images/roc_curve.png)

#### Feature Importance

![Feature Importance](images/feature_importance.png)

#### Test Results by URL Category

| Test Category | Detection Rate | Primary Layer | Detection Reason |
|---------------|---------------|---------------|------------------|
| Adversarial / Typosquatting | 100% | Layer 1 | Typosquatting + character substitution |
| Homoglyph Style | 100% | Layer 1 | Visual character lookalikes |
| IP Address URLs | 100% | Layer 1 | Excessive subdomains |
| Brand Impersonation | 100% | Layer 2 | Brand + phishing keyword |
| Long Phishing URLs | 100% | Layer 2/3 | Brand keyword or ML prediction |
| URL Shortening | 100% | Layer 3 | ML prediction |
| Phishing-like URLs | 100% | Layer 2/3 | Brand keyword or ML prediction |
| Legitimate URLs | 100% correct | Layer 3 | Correct classification |
| Suspicious but Legitimate | Partial | Layer 3 | ML probability threshold |

#### Adversarial Layer Zero-Day Detection Results

| Attack Type | Example URL | Detected | Detection Reason |
|------------|-------------|----------|------------------|
| Leet-speak | g00gle.com | Yes | Character substitution |
| Leet-speak | paypa1.com | Yes | Character substitution |
| Repeated chars | gooooogle.com | Yes | Repeated characters |
| Repeated chars | faceeboook.com | Yes | Typosquatting |
| Punycode | xn--pple-43d.com | Yes | Punycode encoding |
| Domain shadowing | paypal.com.secure-login.com | Yes | Excessive subdomains |

#### Comparison with State-of-the-Art Methods

| Method | Accuracy | AUC-ROC | Zero-Day Capable | Interpretable |
|-------|---------|---------|------------------|---------------|
| PhishGuard (This Work) | 96.65% | 0.9943 | Yes | Yes |
| WebPhish (CNN) | 98.10% | N/A | No | No |
| Boolean Algebra | 89.10% | 0.91 | No | Yes |
| RNN-LSTM | 98.70% | N/A | No | No |
| GRU | 99.18% | N/A | No | No |
| URLNet | 97.20% | N/A | No | No |
| Reference Random Forest | 96.70% | N/A | No | No |

---

## Tools & Technologies

**Programming Language:**

* Python

**Libraries Used:**

* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Python-Levenshtein

---

## How to Run the Project

Clone the repository:
```bash
git clone https://github.com/amrita-tifac-cyber-course-repo/2025-26-EVEN-MTech-24CY731-DMML
```

Navigate to project folder:

```bash
cd 2025-26-EVEN-MTech-24CY731-DMML/projects/DMML#06/src
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Conclusion
This project successfully developed a multi-layer zero-day phishing URL detection system using machine learning and adversarial URL analysis techniques.
