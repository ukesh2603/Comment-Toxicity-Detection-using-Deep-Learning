# Comment Toxicity Detection using Deep Learning

##  Project Overview

Comment Toxicity Detection is a Deep Learning and Natural Language Processing (NLP) project designed to automatically identify toxic comments in conversations. The application analyzes user comments and predicts different types of toxicity, helping online platforms maintain a safe and respectful environment.

The project utilizes text preprocessing techniques, deep learning models and an interactive Streamlit dashboard for real time predictions and bulk comment analysis.

---

## Problem Statement

Online communities and social media platforms have become an essential part of daily communication. However, the increasing number of toxic comments including harassment, hate speech, insults, threats and abusive language creates challenges in maintaining healthy online discussions.

Manual moderation is time consuming and inefficient for large scale platforms. Therefore, an automated toxicity detection system is required to identify harmful comments in real time and assist moderators in taking appropriate actions.

This project develops a Deep Learning based toxicity detection model capable of classifying comments into multiple toxicity categories, enabling safer digital communities.

---

# Business Use Cases

-  Social Media Platforms
-  Online Discussion Forums
-  Content Moderation Services
-  Brand Safety & Advertisement Monitoring
-  E-learning Platforms
-  News & Media Websites
-  Gaming Communities
-  Enterprise Communication Platforms

---

#  Dataset

The dataset contains thousands of user comments labeled into multiple toxicity categories.

### Target Labels

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

Each comment may belong to one or multiple categories, making this a **multi-label classification problem**.

---

# Technologies Used

| Category | Technologies |
|-----------|-------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP | NLTK, Regex |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Deployment | Streamlit |
| Model Storage | Pickle (.pkl) |

---

# Project Structure

```
Deep_learning_Comment_Toxicity/
│
├── Data/
│   ├── train.csv
│   └── test.csv
│
├── Data_Loading_and_Exploration.ipynb
├── Data_Preprocessing.ipynb
│
│
├── Model/
│   ├── Model_Development.ipynb
│   ├── model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Project Workflow

```
Dataset
    │
    ▼
Text Cleaning
    │
    ▼
Text Preprocessing
    │
    ▼
Tokenization
    │
    ▼
Padding & Vectorization
    │
    ▼
Deep Learning Model
(LSTM)
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Save Trained Model
    │
    ▼
Streamlit Web Application
    │
    ▼
Real-Time Toxicity Prediction
```

---

# Data Preprocessing

The following preprocessing techniques were applied:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Remove numbers
- Remove special characters
- Expand contractions
- Remove extra whitespaces
- Tokenization
- Padding sequences
- Vocabulary generation

---

# Exploratory Data Analysis

Performed detailed EDA including:

- Distribution of toxicity labels
- Comment length analysis
- Word frequency visualization
- Multi label distribution

---

# Model Development

A Deep Learning model was developed for multi label text classification.


### Training Steps

- Train Test Split
- Tokenization
- Sequence Padding
- Embedding Layer
- LSTM Network
- Binary Cross Entropy Loss
- Adam Optimizer
- Epoch wise Training
- Model Evaluation

---

# Model Performance

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Streamlit Application

The project includes an interactive Streamlit dashboard.


### Single Comment Prediction

Users can type a comment and instantly receive toxicity predictions.

Example:

```
Input:

You are the worst person ever.

Prediction:

✔ Toxic
✔ Insult
✘ Threat
✘ Identity Hate
```

---

### Bulk Prediction

Users can upload a CSV file containing multiple comments.

The application automatically:

- Reads CSV
- Predicts toxicity
- Displays results
- Downloads prediction file

---

### Model Performance

Displays:

- Accuracy
- Precision
- Recall
- F1 Score

---

# How to Run

## Clone Repository

```bash
git clone https://github.com/ukesh2603/Comment-Toxicity-Detection-using-Deep-Learning
```

---

## Navigate to Project

```bash
cd Deep_learning_Comment_Toxicity
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit

```bash
streamlit run app.py
```

---

# Results

The developed application successfully:

- Detects toxic comments in real time
- Supports multi label classification
- Provides instant prediction results
- Performs bulk comment analysis
- Assists moderators in content filtering
- Improves online community safety

---

# Future Enhancements

- BERT based Transformer Model
- Multilingual Toxicity Detection
- REST API Deployment
- Docker Containerization
- Cloud Deployment (AWS/Azure)
- Live Social Media Integration

---

# Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing
- Text Preprocessing
- Deep Learning using PyTorch
- Multi label Classification
- LSTM Networks
- Model Evaluation
- Streamlit Application Development
- Model Deployment

---

## ⭐ If you found this project useful, please consider giving it a Star!