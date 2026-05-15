# ==============================================================================
# TextAnalytica — Text Preprocessing Module
# ==============================================================================
"""
Comprehensive text preprocessing pipeline for NLP tasks.

Covers:
    - Text cleaning (HTML, URLs, mentions, special characters)
    - Tokenization (word and sentence level)
    - Stopword removal
    - Stemming (Porter Stemmer)
    - Lemmatization (WordNet Lemmatizer)
    - N-gram generation (bigrams, trigrams, custom n)
    - Full preprocessing pipeline combining all steps

Author: Venkata Balaji
Course: CSI324 — Text Analytics
"""

import re
import string
from typing import List, Tuple, Optional

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]

for _res in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{_res}" if "punkt" in _res else f"corpora/{_res}" if _res in ("stopwords", "wordnet") else f"taggers/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)


# ==============================================================================
# Text Cleaning
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise.

    Steps:
        1. Convert to lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove @mentions
        5. Remove hashtag symbols (keep the word)
        6. Remove numbers
        7. Remove punctuation
        8. Remove extra whitespace

    Args:
        text: Raw input text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================================================================
# Tokenization
# ==============================================================================

def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into individual words using NLTK word_tokenize.

    Args:
        text: Input text string.

    Returns:
        List of word tokens.
    """
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK sent_tokenize.

    Args:
        text: Input text string.

    Returns:
        List of sentence strings.
    """
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split(".") if s.strip()]


# ==============================================================================
# Stopword Removal
# ==============================================================================

def remove_stopwords(tokens: List[str], extra_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Remove English stopwords from a list of tokens.

    Args:
        tokens: List of word tokens.
        extra_stopwords: Additional custom stopwords to remove.

    Returns:
        Filtered list of tokens without stopwords.
    """
    stop_words = set(stopwords.words("english"))
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    return [token for token in tokens if token.lower() not in stop_words]


# ==============================================================================
# Stemming
# ==============================================================================

_stemmer = PorterStemmer()


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply Porter Stemming to a list of tokens.

    Example:
        ['running', 'happily'] -> ['run', 'happili']

    Args:
        tokens: List of word tokens.

    Returns:
        List of stemmed tokens.
    """
    return [_stemmer.stem(token) for token in tokens]


def stem_text(text: str) -> str:
    """
    Stem all words in a text string.

    Args:
        text: Input text string.

    Returns:
        Text with all words stemmed.
    """
    tokens = tokenize_words(text)
    return " ".join(stem_tokens(tokens))


# ==============================================================================
# Lemmatization
# ==============================================================================

_lemmatizer = WordNetLemmatizer()


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Apply WordNet Lemmatization to a list of tokens.

    Example:
        ['running', 'better', 'mice'] -> ['running', 'better', 'mouse']

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    return [_lemmatizer.lemmatize(token) for token in tokens]


def lemmatize_text(text: str) -> str:
    """
    Lemmatize all words in a text string.

    Args:
        text: Input text string.

    Returns:
        Text with all words lemmatized.
    """
    tokens = tokenize_words(text)
    return " ".join(lemmatize_tokens(tokens))


# ==============================================================================
# N-gram Generation
# ==============================================================================

def generate_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens: List of word tokens.
        n: Size of the n-gram (2 = bigrams, 3 = trigrams, etc.)

    Returns:
        List of n-gram tuples.
    """
    return list(ngrams(tokens, n))


def get_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """Generate bigrams from tokens."""
    return generate_ngrams(tokens, n=2)


def get_trigrams(tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Generate trigrams from tokens."""
    return generate_ngrams(tokens, n=3)


# ==============================================================================
# Full Preprocessing Pipeline
# ==============================================================================

def preprocess_text(
    text: str,
    do_clean: bool = True,
    do_tokenize: bool = True,
    do_remove_stopwords: bool = True,
    do_stem: bool = False,
    do_lemmatize: bool = True,
) -> dict:
    """
    Run the full preprocessing pipeline on input text.

    This function applies cleaning, tokenization, stopword removal,
    and either stemming or lemmatization (or both) to produce a
    comprehensive preprocessing result.

    Args:
        text: Raw input text.
        do_clean: Whether to clean the text.
        do_tokenize: Whether to tokenize.
        do_remove_stopwords: Whether to remove stopwords.
        do_stem: Whether to apply stemming.
        do_lemmatize: Whether to apply lemmatization.

    Returns:
        Dictionary with keys:
            - original: Original text
            - cleaned: Cleaned text
            - tokens: Word tokens
            - tokens_no_stopwords: Tokens after stopword removal
            - stemmed: Stemmed tokens
            - lemmatized: Lemmatized tokens
            - bigrams: List of bigram tuples
            - trigrams: List of trigram tuples
            - processed_text: Final processed text string
    """
    result = {"original": text}

    # Step 1: Cleaning
    cleaned = clean_text(text) if do_clean else text
    result["cleaned"] = cleaned

    # Step 2: Tokenization
    tokens = tokenize_words(cleaned) if do_tokenize else cleaned.split()
    result["tokens"] = tokens

    # Step 3: Stopword removal
    if do_remove_stopwords:
        tokens_filtered = remove_stopwords(tokens)
    else:
        tokens_filtered = tokens
    result["tokens_no_stopwords"] = tokens_filtered

    # Step 4: Stemming
    if do_stem:
        stemmed = stem_tokens(tokens_filtered)
    else:
        stemmed = tokens_filtered
    result["stemmed"] = stemmed

    # Step 5: Lemmatization
    if do_lemmatize:
        lemmatized = lemmatize_tokens(tokens_filtered)
    else:
        lemmatized = tokens_filtered
    result["lemmatized"] = lemmatized

    # Step 6: N-grams (computed on the final filtered tokens)
    final_tokens = lemmatized if do_lemmatize else (stemmed if do_stem else tokens_filtered)
    result["bigrams"] = get_bigrams(final_tokens)
    result["trigrams"] = get_trigrams(final_tokens)

    # Final processed text
    result["processed_text"] = " ".join(final_tokens)

    return result


def preprocess_for_ml(text: str) -> str:
    """
    Quick preprocessing for ML model input.

    Applies: cleaning → tokenization → stopword removal → lemmatization.
    Returns a single string suitable for vectorizer input.

    Args:
        text: Raw text input.

    Returns:
        Preprocessed text string.
    """
    result = preprocess_text(text, do_stem=False, do_lemmatize=True)
    return result["processed_text"]


# ==============================================================================
# Utility: Batch Preprocessing
# ==============================================================================

def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Preprocess a batch of texts for ML pipeline input.

    Args:
        texts: List of raw text strings.

    Returns:
        List of preprocessed text strings.
    """
    return [preprocess_for_ml(t) for t in texts]
