<<<<<<< HEAD
# 🧠 TextAnalytica — End-to-End NLP & Text Analytics Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.x-yellow.svg)](https://huggingface.co/transformers/)

> A production-grade NLP platform built for **CSI324: Text Analytics** — covering text preprocessing, classification, sentiment analysis, topic modeling, NER, summarization, and BERT-based inference, deployed via Streamlit.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Course Outcomes Mapping](#-course-outcomes-mapping)
- [Evaluation Metrics](#-evaluation-metrics)
- [Screenshots](#-screenshots)
- [License](#-license)

---

## 🔍 Overview

**TextAnalytica** is a comprehensive NLP platform that demonstrates industry-standard text analytics capabilities. Built as a modular Python application with a Streamlit web interface, it covers the full spectrum of NLP tasks from basic preprocessing to transformer-based deep learning.

### Key Highlights

- 🔤 **Text Preprocessing Pipeline** — Tokenization, stemming, lemmatization, stopword removal, n-grams
- 📊 **Feature Engineering** — TF-IDF, Bag of Words, Word2Vec embeddings
- 🤖 **ML Classification** — Naive Bayes, Logistic Regression with sklearn Pipelines
- 💬 **Sentiment Analysis** — Multi-model sentiment prediction (ML + BERT)
- 📰 **Topic Modeling** — LDA-based topic discovery with visualization
- 🏷️ **Named Entity Recognition** — spaCy NER with dependency parsing
- 📝 **Text Summarization** — Extractive summarization using sentence scoring
- 🧠 **BERT Classification** — DistilBERT fine-tuned sentiment classifier
- 🌐 **Streamlit UI** — Professional multi-page web application

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT WEB UI                            │
│  ┌──────┐ ┌──────────┐ ┌─────────┐ ┌─────┐ ┌───────────────┐  │
│  │ Home │ │Preprocess│ │ TF-IDF  │ │ NER │ │ Summarization │  │
│  └──────┘ └──────────┘ └─────────┘ └─────┘ └───────────────┘  │
│  ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌────────────┐   │
│  │Sentiment │ │Spam Detection│ │Topic Model│ │    BERT     │   │
│  └──────────┘ └──────────────┘ └───────────┘ └────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      ENGINE LAYER                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │
│  │ preprocessor  │ │  ml_models   │ │   nlp_tasks          │   │
│  │  .py          │ │  .py         │ │   .py                │   │
│  │ ─────────── │ │ ─────────── │ │ ────────────────── │   │
│  │ Tokenization  │ │ NaiveBayes   │ │ NER (spaCy)        │   │
│  │ Stemming      │ │ LogRegression│ │ Dep. Parsing       │   │
│  │ Lemmatization │ │ TF-IDF       │ │ Summarization      │   │
│  │ Stopwords     │ │ BoW          │ │ Topic Modeling     │   │
│  │ N-grams       │ │ Word2Vec     │ │ BERT Classifier    │   │
│  └──────────────┘ └──────────────┘ └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      DATA LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Sentiment140 Dataset (1.6M tweets)                      │  │
│  │  training.1600000.processed.noemoticon.csv               │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    MODEL ARTIFACTS                              │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ nb_model │ │ lr_model  │ │ tfidf_   │ │  word2vec_     │  │
│  │ .joblib  │ │ .joblib   │ │vectorizer│ │  model.bin     │  │
│  └──────────┘ └───────────┘ └──────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| # | Feature | Module | Status |
|---|---------|--------|--------|
| 1 | Text Preprocessing | `preprocessor.py` | ✅ |
| 2 | Tokenization | `preprocessor.py` | ✅ |
| 3 | Stopword Removal | `preprocessor.py` | ✅ |
| 4 | Stemming (Porter) | `preprocessor.py` | ✅ |
| 5 | Lemmatization (WordNet) | `preprocessor.py` | ✅ |
| 6 | N-gram Generation | `preprocessor.py` | ✅ |
| 7 | Bag of Words | `ml_models.py` | ✅ |
| 8 | TF-IDF Vectorization | `ml_models.py` | ✅ |
| 9 | Document Similarity | `ml_models.py` | ✅ |
| 10 | Naive Bayes Classifier | `ml_models.py` | ✅ |
| 11 | Logistic Regression | `ml_models.py` | ✅ |
| 12 | Sentiment Analysis | `ml_models.py` | ✅ |
| 13 | Spam/Review Classification | `ml_models.py` | ✅ |
| 14 | Topic Modeling (LDA) | `nlp_tasks.py` | ✅ |
| 15 | Word2Vec Embeddings | `nlp_tasks.py` | ✅ |
| 16 | spaCy NER | `nlp_tasks.py` | ✅ |
| 17 | Dependency Parsing | `nlp_tasks.py` | ✅ |
| 18 | BERT Sentiment Classification | `nlp_tasks.py` | ✅ |
| 19 | Extractive Summarization | `nlp_tasks.py` | ✅ |
| 20 | Streamlit Deployment UI | `app.py` | ✅ |

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.9+ |
| Data | pandas, numpy |
| NLP | NLTK, spaCy, gensim |
| ML | scikit-learn |
| Deep Learning | Transformers (HuggingFace), PyTorch |
| Visualization | matplotlib, seaborn, wordcloud, plotly |
| Web UI | Streamlit |
| Model Storage | joblib |

---

## 📁 Project Structure

```
IBM CA CSI324/
│
├── app.py                      # Streamlit main application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source modules
│   ├── __init__.py
│   ├── preprocessor.py         # Text preprocessing pipeline
│   ├── ml_models.py            # ML training & inference
│   └── nlp_tasks.py            # NER, LDA, BERT, Summarization
│
├── training/                   # Model training scripts
│   └── train_models.py         # Train and save all models
│
├── models/                     # Saved model artifacts
│   ├── nb_pipeline.joblib
│   ├── lr_pipeline.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── lda_model.joblib
│   └── word2vec.model
│
├── data/                       # Dataset (symlinked or copied)
│   └── (dataset placed here)
│
├── notebooks/                  # Jupyter notebooks (optional)
│   └── exploration.ipynb
│
└── assets/                     # Static assets for Streamlit
    └── banner.png
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- ~4 GB RAM minimum (for BERT inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/textanalytica-nlp.git
cd textanalytica-nlp
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data & spaCy Model

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
python -m spacy download en_core_web_sm
```

### Step 5: Place the Dataset

Make sure `training.1600000.processed.noemoticon.csv` is in the project root directory.

### Step 6: Train Models

```bash
python training/train_models.py
```

This will train all ML models and save them to the `models/` directory.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage

### Text Preprocessing
Enter any text to see tokenization, stemming, lemmatization, and n-gram generation in action.

### TF-IDF Analysis
Input multiple documents to compute TF-IDF vectors and document similarity scores.

### Sentiment Analysis
Predict sentiment using Naive Bayes, Logistic Regression, or DistilBERT.

### Spam Detection
Classify text as spam or ham using the trained ML pipeline.

### Topic Modeling
Discover latent topics from a collection of documents using LDA.

### NER & Dependency Parsing
Extract named entities and visualize syntactic dependencies with spaCy.

### Summarization
Generate extractive summaries from long text documents.

### BERT Prediction
Use DistilBERT for high-accuracy sentiment classification.

---

## ☁️ Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: TextAnalytica NLP Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/textanalytica-nlp.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add `packages.txt`** (if needed for system dependencies):
```
libgomp1
```

> **Note:** For Streamlit Cloud, the dataset must be small or hosted externally. The app supports both local and demo modes.

---

## 📚 Course Outcomes Mapping

| Course Outcome | Implementation |
|---------------|---------------|
| Fundamentals of NLP | Preprocessing module, tokenization, stemming, lemmatization |
| TF-IDF, BoW, Embeddings | TF-IDF vectorizer, BoW, Word2Vec |
| Text Classification | Naive Bayes, Logistic Regression pipelines |
| Sentiment Analysis | ML + BERT-based sentiment classifiers |
| Topic Modeling | LDA with gensim |
| spaCy NLP Tasks | NER, dependency parsing |
| Transformers & BERT | DistilBERT sentiment classification |
| Real-world Applications | Full Streamlit deployment |

---

## 📊 Evaluation Metrics

The training pipeline generates the following metrics for each model:

- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1-Score** (weighted)
- **Confusion Matrix**
- **Classification Report**

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Venkata Balaji**
B.Tech Big Data Analytics
CSI324: Text Analytics — University Project

---

*Built with ❤️ using Python, scikit-learn, spaCy, Transformers, and Streamlit*
=======
# AI-Text-Analytics-Sentiment-Analysis-Platform
Industry-level NLP and Text Analytics platform using Python, TF-IDF, Machine Learning, BERT, spaCy, and Streamlit deployment.
>>>>>>> df798a764877ddbecbe781d55f4d8ef4a1d5b23f
