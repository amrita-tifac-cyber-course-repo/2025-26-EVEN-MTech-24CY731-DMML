2025-26-EVEN-MTech-24CY731-DMML
2025-26 EVEN MTech 24CY731 DMML
🔐 AI & ML/DL Applications in Cybersecurity
1. Project Title
Phishing Email Detection Using Logistic Regression

2. Team Details

Team Number: 18
Members:

CB.SC.P2CYS25001 – Amgad R




3. Problem Statement
Phishing emails deceive recipients into disclosing sensitive credentials, financial data, and personal information by impersonating trusted entities. Existing rule-based and signature-driven filters fail to generalise to unseen phishing campaigns and produce only binary outputs with no confidence information. This project addresses automated phishing email detection using a machine learning pipeline that classifies emails as legitimate or phishing based solely on text content, and produces a calibrated three-tier risk score (High / Medium / Low) to enable priority-driven threat response.

4. Objectives

Build a text-only phishing email classifier using Logistic Regression with TF-IDF feature extraction that achieves high recall on the phishing class
Consolidate a multi-source dataset of 13,565 emails from four publicly available corpora covering credential theft, Nigerian fraud, spam, and legitimate email categories
Design and implement a custom NLP preprocessing pipeline including URL tokenisation, rule-based lemmatisation, and vocabulary filtering
Extend the model's probability output into an interpretable three-tier risk scoring system (High ≥ 75%, Medium ≥ 45%, Low < 45%) for graduated threat assessment


5. Dataset Details

Dataset Name: Phishing Email Dataset (Ling-Spam, Nazario, Nigerian Fraud, SpamAssassin)
Source (Link): https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
Features Description:

subject — Email subject line (concatenated with body as input text)
body — Full email body text
label — Binary class label: 0 = Legitimate, 1 = Phishing


Dataset Size: 13,565 emails total — 6,492 Legitimate / 7,073 Phishing


6. Methodology

Techniques Used:

Logistic Regression (primary classifier, C=0.1, L2 regularisation)


Model Architecture: Classical ML pipeline (no deep learning)
Workflow:

Data Collection — Load and merge four CSV datasets (Ling, Nazario, Nigerian Fraud, SpamAssassin); concatenate subject + body into a single text field
Data Preprocessing — Lowercase, replace URLs/emails/numbers with placeholder tokens, remove punctuation, apply rule-based lemmatisation, drop tokens < 3 characters
Feature Engineering — TF-IDF vectorisation with ngram_range=(1,2), max_features=15000, sublinear_tf=True, min_df=2, max_df=0.95
Model Training — 80/20 stratified train-test split; Logistic Regression trained on 10,852 emails
Model Evaluation — Classification report (precision, recall, F1), confusion matrix, ROC-AUC, 5-fold cross-validation, risk scorer validation on test cases




7. Tools & Technologies

Programming Language: Python 3.12
Libraries:

Scikit-learn (TfidfVectorizer, LogisticRegression, cross_val_score, metrics)
Pandas, NumPy
Matplotlib, Seaborn
Re, String (preprocessing)
