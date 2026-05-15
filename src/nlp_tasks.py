# ==============================================================================
# TextAnalytica — Advanced NLP Tasks Module
# ==============================================================================
"""
Advanced NLP capabilities including NER, topic modeling, summarization,
Word2Vec embeddings, and BERT-based classification.

Covers:
    - spaCy Named Entity Recognition (NER)
    - spaCy Dependency Parsing
    - Topic Modeling with LDA (gensim)
    - Word2Vec Embeddings (gensim)
    - Extractive Text Summarization
    - BERT / DistilBERT Sentiment Classification

Author: Venkata Balaji
Course: CSI324 — Text Analytics
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------
import spacy
from spacy import displacy

def _load_spacy_model():
    """Load spaCy English model, downloading if necessary."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("📥 Downloading spaCy en_core_web_sm model...")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

_nlp = None

def get_spacy_nlp():
    """Get the spaCy NLP model (lazy-loaded singleton)."""
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp


# ==============================================================================
# Named Entity Recognition (NER) — spaCy
# ==============================================================================

def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text using spaCy NER.

    Args:
        text: Input text string.

    Returns:
        List of dictionaries with 'text', 'label', 'start', 'end' keys.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_) or ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def render_ner_html(text: str) -> str:
    """
    Render NER visualization as HTML using spaCy displaCy.

    Args:
        text: Input text string.

    Returns:
        HTML string with NER highlights.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=False)
    return html


# ==============================================================================
# Dependency Parsing — spaCy
# ==============================================================================

def parse_dependencies(text: str) -> List[Dict[str, str]]:
    """
    Parse syntactic dependencies using spaCy.

    Args:
        text: Input text string.

    Returns:
        List of dicts with 'token', 'dep', 'head', 'pos', 'tag' keys.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    deps = []
    for token in doc:
        deps.append({
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "children": [child.text for child in token.children],
        })
    return deps


def render_dep_html(text: str) -> str:
    """
    Render dependency parse as SVG using spaCy displaCy.

    Args:
        text: Input text string.

    Returns:
        SVG string of dependency parse tree.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    # Use only first sentence for readability
    sentences = list(doc.sents)
    if sentences:
        svg = displacy.render(sentences[0], style="dep", page=False, options={"compact": True})
    else:
        svg = displacy.render(doc, style="dep", page=False, options={"compact": True})
    return svg


# ==============================================================================
# Topic Modeling — LDA (gensim)
# ==============================================================================

def train_lda_model(
    texts: List[List[str]],
    num_topics: int = 5,
    passes: int = 15,
    random_state: int = 42,
):
    """
    Train an LDA topic model using gensim.

    Args:
        texts: List of tokenized documents (list of list of strings).
        num_topics: Number of topics to discover.
        passes: Number of training passes.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lda_model, corpus, dictionary).
    """
    from gensim import corpora
    from gensim.models import LdaMulticore

    # Build dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    # Filter extremes
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # Train LDA
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        workers=2,
    )

    return lda_model, corpus, dictionary


def get_topics(lda_model, num_words: int = 10) -> List[Dict[str, Any]]:
    """
    Extract topics and their top words from a trained LDA model.

    Args:
        lda_model: Trained gensim LDA model.
        num_words: Number of top words per topic.

    Returns:
        List of dicts with 'topic_id', 'words', and 'weights'.
    """
    topics = []
    for idx, topic in lda_model.print_topics(num_words=num_words):
        # Parse the topic string
        word_weight_pairs = re.findall(r'([\d.]+)\*"(\w+)"', topic)
        words = [w for _, w in word_weight_pairs]
        weights = [float(w) for w, _ in word_weight_pairs]
        topics.append({
            "topic_id": idx,
            "words": words,
            "weights": weights,
            "raw": topic,
        })
    return topics


def get_document_topics(lda_model, corpus, doc_index: int = 0) -> List[Tuple[int, float]]:
    """
    Get topic distribution for a specific document.

    Args:
        lda_model: Trained LDA model.
        corpus: Gensim corpus.
        doc_index: Index of the document.

    Returns:
        List of (topic_id, probability) tuples.
    """
    return lda_model.get_document_topics(corpus[doc_index])


# ==============================================================================
# Word2Vec Embeddings — gensim
# ==============================================================================

def train_word2vec(
    tokenized_texts: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20,
    sg: int = 1,
):
    """
    Train a Word2Vec model using gensim.

    Args:
        tokenized_texts: List of tokenized documents.
        vector_size: Dimensionality of word vectors.
        window: Context window size.
        min_count: Minimum word frequency.
        epochs: Training epochs.
        sg: 1 for Skip-gram, 0 for CBOW.

    Returns:
        Trained Word2Vec model.
    """
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=sg,
        workers=4,
    )
    return model


def get_similar_words(w2v_model, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Find most similar words using Word2Vec.

    Args:
        w2v_model: Trained Word2Vec model.
        word: Query word.
        top_n: Number of similar words to return.

    Returns:
        List of (word, similarity_score) tuples.
    """
    try:
        return w2v_model.wv.most_similar(word, topn=top_n)
    except KeyError:
        return []


def get_word_vector(w2v_model, word: str) -> Optional[np.ndarray]:
    """
    Get the vector representation of a word.

    Args:
        w2v_model: Trained Word2Vec model.
        word: Query word.

    Returns:
        Numpy array of the word vector, or None if not in vocabulary.
    """
    try:
        return w2v_model.wv[word]
    except KeyError:
        return None


# ==============================================================================
# Extractive Text Summarization
# ==============================================================================

def extractive_summarize(text: str, num_sentences: int = 3) -> Dict[str, Any]:
    """
    Generate an extractive summary by scoring sentences based on
    word frequency and selecting the top-ranked sentences.

    Algorithm:
        1. Tokenize text into sentences
        2. Compute word frequencies (excluding stopwords)
        3. Score each sentence by sum of word frequencies
        4. Select top-N sentences (preserving original order)

    Args:
        text: Input text to summarize.
        num_sentences: Number of sentences in summary.

    Returns:
        Dictionary with 'summary', 'scores', and 'sentences'.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)

    # Get sentences
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(sentences) <= num_sentences:
        return {
            "summary": text,
            "sentences": sentences,
            "scores": {s: 1.0 for s in sentences},
        }

    # Compute word frequencies (excluding stopwords and punctuation)
    word_freq = Counter()
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text.strip():
            word_freq[token.text.lower()] += 1

    # Normalize frequencies
    if word_freq:
        max_freq = max(word_freq.values())
        word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    # Score sentences
    sentence_scores = {}
    for sent in sentences:
        sent_doc = nlp(sent)
        score = sum(word_freq.get(token.text.lower(), 0) for token in sent_doc
                     if not token.is_stop and not token.is_punct)
        # Normalize by sentence length to avoid bias toward long sentences
        length = len([t for t in sent_doc if not t.is_punct])
        if length > 0:
            score = score / length
        sentence_scores[sent] = round(score, 4)

    # Select top sentences preserving original order
    ranked = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences = set(s for s, _ in ranked[:num_sentences])
    summary_sentences = [s for s in sentences if s in top_sentences]

    return {
        "summary": " ".join(summary_sentences),
        "sentences": sentences,
        "scores": sentence_scores,
    }


# ==============================================================================
# BERT / DistilBERT Sentiment Classification
# ==============================================================================

_bert_pipeline = None


def get_bert_pipeline():
    """
    Load the DistilBERT sentiment analysis pipeline (lazy-loaded singleton).

    Uses HuggingFace's 'distilbert-base-uncased-finetuned-sst-2-english'
    pre-trained model for sentiment classification.

    Returns:
        HuggingFace pipeline for sentiment analysis.
    """
    global _bert_pipeline
    if _bert_pipeline is None:
        print("🤖 Loading DistilBERT sentiment model...")
        from transformers import pipeline as hf_pipeline
        _bert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # CPU
        )
        print("✅ DistilBERT model loaded successfully.")
    return _bert_pipeline


def bert_predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment using DistilBERT.

    Args:
        text: Input text string.

    Returns:
        Dictionary with 'label' and 'score'.
    """
    pipe = get_bert_pipeline()
    # Truncate to 512 tokens (BERT limit)
    result = pipe(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4),
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
    }


def bert_predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Batch sentiment prediction using DistilBERT.

    Args:
        texts: List of text strings.

    Returns:
        List of prediction dictionaries.
    """
    pipe = get_bert_pipeline()
    truncated = [t[:512] for t in texts]
    results = pipe(truncated, batch_size=16)
    return [
        {"label": r["label"], "score": round(r["score"], 4)}
        for r in results
    ]


# ==============================================================================
# POS Tagging — spaCy
# ==============================================================================

def pos_tag(text: str) -> List[Dict[str, str]]:
    """
    Part-of-speech tagging using spaCy.

    Args:
        text: Input text string.

    Returns:
        List of dicts with 'token', 'pos', 'tag', 'explanation'.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    return [
        {
            "token": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "explanation": spacy.explain(token.tag_) or token.tag_,
        }
        for token in doc
    ]
