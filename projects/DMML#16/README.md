# Deep Learning Based Detection of SMS Phishing Attack

## 1. Project Title
Deep Learning Based Detection of SMS Phishing Attack

## 2. Team Details
- **Team Number:**  16
- **Members:**
  - CB.SC.P2CYS25013 – Krishnapriya S
  - CB.SC.P2CYS25019 – Nidhisha S Dharan

## 3. Problem Statement
Smishing (SMS Phishing) is a growing cybersecurity threat in which attackers send 
fraudulent text messages to trick users into revealing sensitive information such as 
passwords, bank details, or personal data. Existing detection systems primarily perform 
binary classification, identifying a message as spam or legitimate, without providing 
any information about the nature of the attack. This project addresses the need for a 
more informative and practical smishing detection system.

## 4. Objectives
- To develop a two-stage hierarchical smishing detection model capable of both 
  detecting and classifying phishing attacks by attack type
- To classify detected phishing messages into phone-based, URL-based, or 
  email-based attack categories
- To deploy the model as a lightweight on-device Android application for real-time 
  detection
- To evaluate the system on a real-world dataset combining two publicly available 
  smishing datasets

---

## 5. Dataset Details
- **Dataset Name:** Combined SMS Phishing Dataset
- **Source:**
  - Mendeley SMS Phishing Collection 
  - SmishTank 
- **Features Description:**
  - TEXT — Raw content of the SMS message
  - URL — Indicates presence of a URL in the message
  - EMAIL — Indicates presence of an email address in the message
  - PHONE — Indicates presence of a phone number in the message
- **Dataset Size:** Approximately 9,300 SMS messages

---

## 6. Methodology
- **Techniques Used:**
  - DistilBERT (transformer-based language model)
  - Two-stage cascaded classification
  - Multi-task learning
  - AdamW optimiser
  - Weighted Random Sampler for class balancing

- **Model Architecture:**
  - Shared DistilBERT backbone
  - Stage 1 classification head: 768 → 128 → 2 classes (Safe / Phishing)
  - Stage 2 classification head: 768 → 256 → 64 → 3 classes (Phone / URL / Email)

- **Workflow:**
  1. Data Collection
  2. Data Preprocessing (stripping, lowercase conversion, binary encoding)
  3. Label Assignment (S1 and S2 labels)
  4. Data Splitting and Class Balancing (80/10/10)
  5. DistilBERT Tokenization
  6. Model Training (6 epochs, batch size 32)
  7. Model Evaluation
  8. Android Application Deployment

## 7. Tools & Technologies
- **Programming Language:** Python, Java (Android)
- **Libraries:**
  - PyTorch
  - HuggingFace Transformers
  - Scikit-learn
  - Pandas, NumPy, Matplotlib
