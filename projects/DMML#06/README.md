# DMML

# Zero-Day Phishing URL Detection Using Machine Learning and Adversarial URL Analysis

## 1. Project Title

Zero-Day Phishing URL Detection Using Machine Learning and Adversarial URL Analysis

## 2. Team Details

Team Number: DMML#06

Members:
CB.SC.P2CYS25007 – Balina Sri Vaishnavi
CB.SC.P2CYS25006 – Atmala Sai Chandra Koushik

## 3. Problem Statement

Traditional phishing detection methods rely on blacklist and signature-based detection methods which cannot detect zero-day phishing URLs. Attackers continuously generate new phishing domains that are not present in blacklist databases. This project aims to detect zero-day phishing URLs using adversarial URL analysis and machine learning techniques.

## 4. Objectives

* Detect zero-day phishing URLs
* Identify adversarial phishing techniques such as typosquatting and homoglyph attacks
* Extract lexical and structural URL features
* Train machine learning models for phishing detection
* Build a multi-layer phishing detection system

## 5. Dataset Details

Dataset Name: Phishing and Legitimate URL Dataset

Source:

* PhishTank – https://phishtank.org/developer_info.php
* Tranco Top 1M – https://tranco-list.eu

Features Description:

* URL Length
* Number of Dots
* Number of Hyphens
* Digit Count
* Subdomain Count
* Suspicious TLD
* Presence of IP Address
* Punycode Detection
* Phishing Keywords

Dataset Size:
Approximately 40,000 URLs

## 6. Methodology

Techniques Used:

* Logistic Regression
* Random Forest

Workflow:

1. Collect phishing URLs
2. Collect legitimate URLs
3. Merge datasets
4. Clean dataset
5. Shuffle dataset
6. Extract features
7. Train machine learning model
8. Detect adversarial URLs
9. Predict phishing URLs
10. Streamlit web application

## 7. Tools & Technologies

Programming Language: Python

Libraries:

* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Streamlit

## 8. How to Run the Project

Clone the Repository:
git clone <repo-link>

Navigate to source folder:
cd DMML#06/src

Install dependencies:
pip install -r requirements.txt

Run the project:
python app.py

## 9. System Architecture

![Architecture](images/system_architecture.png)

## 10. Workflow

![Workflow](images/workflow.png)

## 11. Results

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)
