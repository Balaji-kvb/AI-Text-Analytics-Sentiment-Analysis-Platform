# ==============================================================================
# TextAnalytica — Model Training Script
# ==============================================================================
"""
Train all ML models on the Sentiment140 dataset and save artifacts.

This script:
    1. Loads and preprocesses the Sentiment140 dataset
    2. Trains a Naive Bayes classifier (sklearn Pipeline)
    3. Trains a Logistic Regression classifier (sklearn Pipeline)
    4. Trains a Word2Vec model
    5. Evaluates all models with full metrics
    6. Saves all models and metrics to the models/ directory

Usage:
    python training/train_models.py

Author: Venkata Balaji
Course: CSI324 — Text Analytics
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessor import preprocess_for_ml, tokenize_words, clean_text
from src.ml_models import (
    build_nb_pipeline,
    build_lr_pipeline,
    train_and_evaluate,
    save_model,
    save_metrics,
)
from src.nlp_tasks import train_word2vec


# ==============================================================================
# Configuration
# ==============================================================================

DATASET_PATH = os.path.join(PROJECT_ROOT, "training.1600000.processed.noemoticon.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAMPLE_SIZE = 100000  # Use 100K samples for training speed (increase for production)
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==============================================================================
# Data Loading
# ==============================================================================

def load_sentiment140(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the Sentiment140 dataset.

    The dataset has no header. Columns:
        0 - sentiment (0=negative, 4=positive)
        1 - id
        2 - date
        3 - query
        4 - user
        5 - text

    Args:
        filepath: Path to the CSV file.
        sample_size: Number of samples to use (None = all).

    Returns:
        DataFrame with 'text' and 'sentiment' columns.
    """
    print("📂 Loading Sentiment140 dataset...")
    col_names = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(filepath, encoding="latin-1", header=None, names=col_names)

    # Keep only text and sentiment
    df = df[["text", "sentiment"]].copy()

    # Convert sentiment: 0 → 0 (negative), 4 → 1 (positive)
    df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

    # Drop any NaN
    df.dropna(inplace=True)

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"📊 Using {sample_size:,} samples (out of {len(df):,})")

    print(f"✅ Dataset loaded: {len(df):,} records")
    print(f"   Class distribution:\n{df['sentiment'].value_counts().to_string()}")

    return df


# ==============================================================================
# Preprocessing
# ==============================================================================

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess text column in the dataset.

    Args:
        df: DataFrame with 'text' column.

    Returns:
        DataFrame with additional 'clean_text' column.
    """
    print("\n🔧 Preprocessing text data...")
    start = time.time()

    df["clean_text"] = df["text"].apply(preprocess_for_ml)

    # Remove empty strings after preprocessing
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    elapsed = time.time() - start
    print(f"✅ Preprocessing complete in {elapsed:.1f}s — {len(df):,} valid records")

    return df


# ==============================================================================
# Main Training Pipeline
# ==============================================================================

def main():
    """Run the complete training pipeline."""
    print("\n" + "=" * 70)
    print("  🧠 TextAnalytica — Model Training Pipeline")
    print("=" * 70)

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load Data
    # ------------------------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print("   Please place 'training.1600000.processed.noemoticon.csv' in the project root.")
        sys.exit(1)

    df = load_sentiment140(DATASET_PATH, sample_size=SAMPLE_SIZE)

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------
    df = preprocess_dataset(df)

    # ------------------------------------------------------------------
    # Step 3: Train/Test Split
    # ------------------------------------------------------------------
    print(f"\n📐 Splitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["sentiment"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"].tolist(),
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # Step 4: Train Naive Bayes
    # ------------------------------------------------------------------
    nb_pipeline = build_nb_pipeline()
    nb_pipeline, nb_metrics = train_and_evaluate(
        nb_pipeline, X_train, y_train, X_test, y_test,
        model_name="Naive Bayes (TF-IDF)"
    )
    save_model(nb_pipeline, os.path.join(MODELS_DIR, "nb_pipeline.joblib"))
    save_metrics(nb_metrics, os.path.join(MODELS_DIR, "nb_metrics.json"))

    # ------------------------------------------------------------------
    # Step 5: Train Logistic Regression
    # ------------------------------------------------------------------
    lr_pipeline = build_lr_pipeline()
    lr_pipeline, lr_metrics = train_and_evaluate(
        lr_pipeline, X_train, y_train, X_test, y_test,
        model_name="Logistic Regression (TF-IDF)"
    )
    save_model(lr_pipeline, os.path.join(MODELS_DIR, "lr_pipeline.joblib"))
    save_metrics(lr_metrics, os.path.join(MODELS_DIR, "lr_metrics.json"))

    # ------------------------------------------------------------------
    # Step 6: Train Word2Vec
    # ------------------------------------------------------------------
    print("\n🔤 Training Word2Vec model...")
    start = time.time()
    # Tokenize the training data for Word2Vec
    tokenized_texts = [tokenize_words(clean_text(t)) for t in X_train[:50000]]
    w2v_model = train_word2vec(tokenized_texts, vector_size=100, window=5, epochs=20)
    w2v_path = os.path.join(MODELS_DIR, "word2vec.model")
    w2v_model.save(w2v_path)
    elapsed = time.time() - start
    print(f"✅ Word2Vec trained in {elapsed:.1f}s — Vocabulary: {len(w2v_model.wv):,} words")
    print(f"💾 Word2Vec saved to {w2v_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  📊 Training Summary")
    print("=" * 70)
    print(f"  {'Model':<35} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Naive Bayes (TF-IDF)':<35} {nb_metrics['accuracy']:>10.4f} {nb_metrics['f1_score']:>10.4f}")
    print(f"  {'Logistic Regression (TF-IDF)':<35} {lr_metrics['accuracy']:>10.4f} {lr_metrics['f1_score']:>10.4f}")
    print(f"  {'Word2Vec':<35} {'N/A':>10} {'N/A':>10}")
    print(f"\n  All models saved to: {MODELS_DIR}")
    print("=" * 70)
    print("\n✅ Training pipeline complete! Run 'streamlit run app.py' to start the app.\n")


if __name__ == "__main__":
    main()
