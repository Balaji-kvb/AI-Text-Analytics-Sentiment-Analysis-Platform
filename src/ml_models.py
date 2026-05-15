# ==============================================================================
# TextAnalytica — Machine Learning Models Module
# ==============================================================================
"""
ML training, evaluation, and inference for text classification.

Covers:
    - TF-IDF Vectorization
    - Bag of Words (CountVectorizer)
    - Document Similarity (cosine similarity)
    - Naive Bayes Classifier (sklearn Pipeline)
    - Logistic Regression Classifier (sklearn Pipeline)
    - Model evaluation (accuracy, precision, recall, F1, confusion matrix)
    - Model saving and loading via joblib

Author: Venkata Balaji
Course: CSI324 — Text Analytics
"""

import os
import json
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# TF-IDF Vectorization
# ==============================================================================

def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.95,
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer with specified parameters.

    Args:
        max_features: Maximum number of features.
        ngram_range: Range for n-gram extraction.
        max_df: Maximum document frequency threshold.
        min_df: Minimum document frequency threshold.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
    )


def compute_tfidf(texts: List[str], vectorizer: Optional[TfidfVectorizer] = None):
    """
    Compute TF-IDF matrix for a list of texts.

    Args:
        texts: List of text strings.
        vectorizer: Pre-fitted vectorizer (optional). If None, a new one is created.

    Returns:
        Tuple of (tfidf_matrix, vectorizer, feature_names).
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, vectorizer, feature_names


def get_top_tfidf_features(tfidf_matrix, feature_names, doc_index: int = 0, top_n: int = 10):
    """
    Get top TF-IDF features for a specific document.

    Args:
        tfidf_matrix: TF-IDF sparse matrix.
        feature_names: Array of feature names.
        doc_index: Index of the document to analyze.
        top_n: Number of top features to return.

    Returns:
        List of (feature, score) tuples sorted by score descending.
    """
    row = tfidf_matrix[doc_index].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(row[i], 4)) for i in top_indices if row[i] > 0]


# ==============================================================================
# Bag of Words
# ==============================================================================

def compute_bow(texts: List[str], max_features: int = 5000):
    """
    Compute Bag of Words representation.

    Args:
        texts: List of text strings.
        max_features: Maximum vocabulary size.

    Returns:
        Tuple of (bow_matrix, vectorizer, feature_names).
    """
    vectorizer = CountVectorizer(max_features=max_features)
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return bow_matrix, vectorizer, feature_names


# ==============================================================================
# Document Similarity
# ==============================================================================

def compute_document_similarity(texts: List[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity between documents using TF-IDF.

    Args:
        texts: List of document strings.

    Returns:
        Cosine similarity matrix (numpy array).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)


# ==============================================================================
# Model Evaluation
# ==============================================================================

def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Computes accuracy, precision, recall, F1-score, confusion matrix,
    and a full classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model (for display).

    Returns:
        Dictionary containing all evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics to console."""
    print(f"\n{'='*60}")
    print(f"  {metrics['model_name']} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n  Classification Report:\n{metrics['classification_report']}")
    print(f"{'='*60}\n")


# ==============================================================================
# Naive Bayes Pipeline
# ==============================================================================

def build_nb_pipeline(max_features: int = 15000) -> Pipeline:
    """
    Build a Naive Bayes text classification pipeline.

    Pipeline:
        TF-IDF Vectorizer → Multinomial Naive Bayes

    Args:
        max_features: Maximum TF-IDF features.

    Returns:
        sklearn Pipeline instance.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])


# ==============================================================================
# Logistic Regression Pipeline
# ==============================================================================

def build_lr_pipeline(max_features: int = 15000) -> Pipeline:
    """
    Build a Logistic Regression text classification pipeline.

    Pipeline:
        TF-IDF Vectorizer → Logistic Regression

    Args:
        max_features: Maximum TF-IDF features.

    Returns:
        sklearn Pipeline instance.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


# ==============================================================================
# Train and Evaluate
# ==============================================================================

def train_and_evaluate(
    pipeline: Pipeline,
    X_train: List[str],
    y_train,
    X_test: List[str],
    y_test,
    model_name: str = "Model",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a pipeline and evaluate it on the test set.

    Args:
        pipeline: sklearn Pipeline to train.
        X_train: Training texts.
        y_train: Training labels.
        X_test: Test texts.
        y_test: Test labels.
        model_name: Name for reporting.

    Returns:
        Tuple of (trained_pipeline, metrics_dict).
    """
    print(f"\n🚀 Training {model_name}...")
    pipeline.fit(X_train, y_train)

    print(f"📊 Evaluating {model_name}...")
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name=model_name)
    print_metrics(metrics)

    return pipeline, metrics


# ==============================================================================
# Model Save / Load
# ==============================================================================

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model or pipeline to save.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a saved model from disk.

    Args:
        filepath: Path to the saved model file.

    Returns:
        Loaded model or pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    model = joblib.load(filepath)
    print(f"📦 Model loaded from {filepath}")
    return model


# ==============================================================================
# Prediction Helpers
# ==============================================================================

def predict_sentiment(pipeline: Pipeline, text: str) -> Dict[str, Any]:
    """
    Predict sentiment for a single text using a trained pipeline.

    Args:
        pipeline: Trained classification pipeline.
        text: Input text string.

    Returns:
        Dictionary with 'prediction', 'label', and 'probabilities'.
    """
    prediction = pipeline.predict([text])[0]

    # Get probability if available
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        probabilities = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}

    label = "Positive" if prediction == 1 else "Negative"

    return {
        "prediction": int(prediction),
        "label": label,
        "probabilities": probabilities,
    }


def predict_batch(pipeline: Pipeline, texts: List[str]) -> List[int]:
    """
    Predict labels for a batch of texts.

    Args:
        pipeline: Trained classification pipeline.
        texts: List of text strings.

    Returns:
        List of predicted labels.
    """
    return pipeline.predict(texts).tolist()


# ==============================================================================
# Save Evaluation Metrics to JSON
# ==============================================================================

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Metrics dictionary from evaluate_model().
        filepath: Destination JSON file path.
    """
    serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"📄 Metrics saved to {filepath}")
