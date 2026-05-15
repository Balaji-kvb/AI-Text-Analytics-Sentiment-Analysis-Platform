# ==============================================================================
# TextAnalytica — Streamlit Application
# ==============================================================================
"""
Multi-page Streamlit NLP Platform covering all CSI324 course outcomes.
Author: Venkata Balaji | Course: CSI324 — Text Analytics
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import (
    preprocess_text, clean_text, tokenize_words, tokenize_sentences,
    remove_stopwords, stem_tokens, lemmatize_tokens, get_bigrams, get_trigrams,
    preprocess_for_ml,
)
from src.ml_models import (
    compute_tfidf, get_top_tfidf_features, compute_document_similarity,
    compute_bow, load_model, predict_sentiment,
)
from src.nlp_tasks import (
    extract_entities, render_ner_html, parse_dependencies, render_dep_html,
    extractive_summarize, bert_predict_sentiment, pos_tag,
    train_lda_model, get_topics, get_similar_words,
)

# ==============================================================================
# Page Config & Styling
# ==============================================================================
st.set_page_config(
    page_title="TextAnalytica — NLP Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        color: white; text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; color: #333; font-size: 0.9rem; font-weight: 500; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #667eea; }

    .result-box {
        background: #f8f9ff; border-left: 4px solid #667eea;
        padding: 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }

    .section-header {
        background: linear-gradient(90deg, #667eea22, transparent);
        padding: 0.8rem 1.2rem; border-radius: 8px; margin: 1.5rem 0 1rem;
        border-left: 3px solid #667eea; font-weight: 600; font-size: 1.1rem;
    }

    .tag { display: inline-block; background: #667eea; color: white;
        padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem;
        margin: 0.2rem; font-weight: 500; }
    .tag-green { background: #10b981; }
    .tag-orange { background: #f59e0b; }
    .tag-red { background: #ef4444; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Model Loading (cached)
# ==============================================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

@st.cache_resource
def load_nb_model():
    try: return load_model(os.path.join(MODELS_DIR, "nb_pipeline.joblib"))
    except: return None

@st.cache_resource
def load_lr_model():
    try: return load_model(os.path.join(MODELS_DIR, "lr_pipeline.joblib"))
    except: return None

@st.cache_resource
def load_w2v_model():
    try:
        from gensim.models import Word2Vec
        return Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
    except: return None

def load_metrics(name):
    try:
        with open(os.path.join(MODELS_DIR, f"{name}_metrics.json")) as f:
            return json.load(f)
    except: return None

# ==============================================================================
# Sidebar Navigation
# ==============================================================================
st.sidebar.markdown("## 🧠 TextAnalytica")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Home",
    "🔤 Text Preprocessing",
    "📊 TF-IDF Analysis",
    "💬 Sentiment Analysis",
    "📧 Spam Detection",
    "📰 Topic Modeling",
    "🏷️ NER & Parsing",
    "📝 Summarization",
    "🤖 BERT Prediction",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**CSI324** — Text Analytics")
st.sidebar.markdown("*Venkata Balaji*")

# ==============================================================================
# PAGE: Home
# ==============================================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 TextAnalytica</h1>
        <p>End-to-End NLP & Text Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Platform Features")
    cols = st.columns(3)
    features = [
        ("🔤", "Preprocessing", "Tokenization, stemming, lemmatization, n-grams"),
        ("📊", "TF-IDF & BoW", "Feature extraction and document similarity"),
        ("💬", "Sentiment", "ML + BERT-based sentiment analysis"),
        ("📧", "Classification", "Spam detection with Naive Bayes & Logistic Reg."),
        ("📰", "Topic Modeling", "LDA topic discovery from documents"),
        ("🏷️", "NER & Parsing", "Named entities and dependency trees via spaCy"),
        ("📝", "Summarization", "Extractive text summarization"),
        ("🤖", "BERT", "DistilBERT transformer-based classification"),
        ("📈", "Metrics", "Accuracy, Precision, Recall, F1, Confusion Matrix"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""<div class="metric-card">
                <h3>{icon} {title}</h3><p style="font-size:0.85rem;color:#666;margin-top:0.5rem">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # Show model metrics if available
    st.markdown("### 📊 Model Performance")
    nb_met = load_metrics("nb")
    lr_met = load_metrics("lr")
    if nb_met and lr_met:
        c1, c2 = st.columns(2)
        for met, col, name in [(nb_met, c1, "Naive Bayes"), (lr_met, c2, "Logistic Regression")]:
            with col:
                st.markdown(f"**{name}**")
                mc = st.columns(4)
                for j, (k, v) in enumerate([("Accuracy", met["accuracy"]),
                    ("Precision", met["precision"]), ("Recall", met["recall"]),
                    ("F1-Score", met["f1_score"])]):
                    mc[j].metric(k, f"{v:.4f}")
    else:
        st.info("⚠️ Models not yet trained. Run `python training/train_models.py` first.")

# ==============================================================================
# PAGE: Text Preprocessing
# ==============================================================================
elif page == "🔤 Text Preprocessing":
    st.markdown('<div class="section-header">🔤 Text Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown("Explore tokenization, stemming, lemmatization, stopword removal, and n-gram generation.")

    text_input = st.text_area("Enter text to preprocess:",
        value="The quick brown foxes were running happily through the beautiful gardens in New York City!",
        height=120)

    c1, c2, c3 = st.columns(3)
    do_stem = c1.checkbox("Stemming", value=True)
    do_lemma = c2.checkbox("Lemmatization", value=True)
    do_stop = c3.checkbox("Remove Stopwords", value=True)

    if st.button("🚀 Process Text", type="primary"):
        result = preprocess_text(text_input, do_stem=do_stem, do_lemmatize=do_lemma, do_remove_stopwords=do_stop)

        tabs = st.tabs(["Cleaned", "Tokens", "Stemmed", "Lemmatized", "N-grams"])
        with tabs[0]:
            st.markdown(f'<div class="result-box"><b>Original:</b> {result["original"]}<br><br><b>Cleaned:</b> {result["cleaned"]}</div>', unsafe_allow_html=True)
        with tabs[1]:
            col1, col2 = st.columns(2)
            col1.markdown("**All Tokens**")
            col1.write(result["tokens"])
            col2.markdown("**After Stopword Removal**")
            col2.write(result["tokens_no_stopwords"])
        with tabs[2]:
            st.write(result["stemmed"])
        with tabs[3]:
            st.write(result["lemmatized"])
        with tabs[4]:
            col1, col2 = st.columns(2)
            col1.markdown("**Bigrams**")
            col1.write([" ".join(b) for b in result["bigrams"]])
            col2.markdown("**Trigrams**")
            col2.write([" ".join(t) for t in result["trigrams"]])

        st.markdown("**Final Processed Text:**")
        st.code(result["processed_text"])

# ==============================================================================
# PAGE: TF-IDF Analysis
# ==============================================================================
elif page == "📊 TF-IDF Analysis":
    st.markdown('<div class="section-header">📊 TF-IDF & Document Similarity</div>', unsafe_allow_html=True)

    st.markdown("Enter multiple documents (one per line) to compute TF-IDF vectors and similarity.")
    docs_input = st.text_area("Documents (one per line):",
        value="Natural language processing is a field of artificial intelligence.\n"
              "Machine learning algorithms can process and understand human language.\n"
              "Deep learning models like BERT have revolutionized NLP tasks.\n"
              "Text analytics involves extracting insights from textual data.",
        height=150)

    if st.button("📊 Compute TF-IDF", type="primary"):
        docs = [d.strip() for d in docs_input.strip().split("\n") if d.strip()]
        if len(docs) < 2:
            st.warning("Please enter at least 2 documents.")
        else:
            tfidf_matrix, vectorizer, features = compute_tfidf(docs)
            tabs = st.tabs(["TF-IDF Scores", "Top Features", "Document Similarity", "Bag of Words"])

            with tabs[0]:
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=features,
                    index=[f"Doc {i+1}" for i in range(len(docs))])
                # Show only non-zero columns
                nonzero = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]
                st.dataframe(nonzero.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

            with tabs[1]:
                for i in range(len(docs)):
                    top = get_top_tfidf_features(tfidf_matrix, features, i, 8)
                    st.markdown(f"**Doc {i+1}:** {docs[i][:80]}...")
                    if top:
                        st.dataframe(pd.DataFrame(top, columns=["Feature", "Score"]), use_container_width=True)

            with tabs[2]:
                sim = compute_document_similarity(docs)
                labels = [f"Doc {i+1}" for i in range(len(docs))]
                sim_df = pd.DataFrame(sim, index=labels, columns=labels)
                st.dataframe(sim_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)

            with tabs[3]:
                bow_matrix, bow_vec, bow_features = compute_bow(docs)
                bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_features,
                    index=[f"Doc {i+1}" for i in range(len(docs))])
                nonzero_bow = bow_df.loc[:, (bow_df != 0).any(axis=0)]
                st.dataframe(nonzero_bow, use_container_width=True)

# ==============================================================================
# PAGE: Sentiment Analysis
# ==============================================================================
elif page == "💬 Sentiment Analysis":
    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)

    nb_model = load_nb_model()
    lr_model = load_lr_model()

    text_input = st.text_area("Enter text for sentiment analysis:",
        value="This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.",
        height=100)

    if st.button("🔍 Analyze Sentiment", type="primary"):
        cols = st.columns(3 if nb_model else 1)

        if nb_model:
            with cols[0]:
                res = predict_sentiment(nb_model, preprocess_for_ml(text_input))
                color = "tag-green" if res["label"] == "Positive" else "tag-red"
                st.markdown(f'**Naive Bayes**<br><span class="tag {color}">{res["label"]}</span>', unsafe_allow_html=True)
                if res["probabilities"]:
                    st.json(res["probabilities"])

        if lr_model:
            with cols[1]:
                res = predict_sentiment(lr_model, preprocess_for_ml(text_input))
                color = "tag-green" if res["label"] == "Positive" else "tag-red"
                st.markdown(f'**Logistic Regression**<br><span class="tag {color}">{res["label"]}</span>', unsafe_allow_html=True)
                if res["probabilities"]:
                    st.json(res["probabilities"])

        col_bert = cols[2] if nb_model else cols[0]
        with col_bert:
            with st.spinner("Loading BERT..."):
                res = bert_predict_sentiment(text_input)
                color = "tag-green" if res["label"] == "POSITIVE" else "tag-red"
                st.markdown(f'**DistilBERT**<br><span class="tag {color}">{res["label"]}</span><br>Confidence: {res["score"]:.4f}', unsafe_allow_html=True)

        if not nb_model:
            st.info("Train ML models with `python training/train_models.py` for NB & LR predictions.")

# ==============================================================================
# PAGE: Spam Detection
# ==============================================================================
elif page == "📧 Spam Detection":
    st.markdown('<div class="section-header">📧 Spam / Review Classification</div>', unsafe_allow_html=True)
    st.markdown("Uses the trained sentiment classifiers to detect spam-like or negative content.")

    nb_model = load_nb_model()
    lr_model = load_lr_model()

    if not nb_model:
        st.warning("⚠️ Models not trained yet. Run `python training/train_models.py`")
    else:
        text_input = st.text_area("Enter text to classify:",
            value="Congratulations! You've won a $1000 gift card. Click here to claim now!!!",
            height=100)

        if st.button("🔍 Classify", type="primary"):
            processed = preprocess_for_ml(text_input)
            c1, c2 = st.columns(2)
            for model, col, name in [(nb_model, c1, "Naive Bayes"), (lr_model, c2, "Logistic Regression")]:
                if model:
                    with col:
                        res = predict_sentiment(model, processed)
                        label = "Likely Spam / Negative" if res["prediction"] == 0 else "Legitimate / Positive"
                        color = "tag-red" if res["prediction"] == 0 else "tag-green"
                        st.markdown(f'**{name}**<br><span class="tag {color}">{label}</span>', unsafe_allow_html=True)
                        if res["probabilities"]:
                            st.json(res["probabilities"])

        # Batch classification
        st.markdown("---")
        st.markdown("**Batch Classification**")
        batch_input = st.text_area("Enter multiple texts (one per line):",
            value="I love this product, works great!\nFREE money!!! Click NOW to win\nThe service was terrible and rude\nGreat experience, highly recommend",
            height=120, key="batch")

        if st.button("📊 Classify Batch"):
            texts = [t.strip() for t in batch_input.split("\n") if t.strip()]
            processed = [preprocess_for_ml(t) for t in texts]
            results = []
            for i, (orig, proc) in enumerate(zip(texts, processed)):
                res = predict_sentiment(nb_model, proc)
                results.append({"Text": orig[:80], "Prediction": res["label"],
                    "Confidence": max(res["probabilities"].values()) if res["probabilities"] else "N/A"})
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ==============================================================================
# PAGE: Topic Modeling
# ==============================================================================
elif page == "📰 Topic Modeling":
    st.markdown('<div class="section-header">📰 Topic Modeling (LDA)</div>', unsafe_allow_html=True)

    docs_input = st.text_area("Enter documents (one per line):",
        value="Machine learning is transforming the healthcare industry with predictive analytics.\n"
              "Natural language processing enables computers to understand human language.\n"
              "Deep learning neural networks achieve state of the art results in computer vision.\n"
              "Data science combines statistics programming and domain expertise.\n"
              "Artificial intelligence is being applied to autonomous driving technology.\n"
              "The stock market uses algorithmic trading powered by machine learning.\n"
              "Healthcare AI can detect diseases from medical images with high accuracy.\n"
              "Chatbots use NLP to provide customer service and support.\n"
              "Recommendation systems use collaborative filtering and deep learning.\n"
              "Computer vision models can identify objects in images and videos.",
        height=200)

    num_topics = st.slider("Number of Topics", 2, 10, 3)

    if st.button("🔍 Discover Topics", type="primary"):
        docs = [d.strip() for d in docs_input.split("\n") if d.strip()]
        if len(docs) < 3:
            st.warning("Enter at least 3 documents.")
        else:
            tokenized = [tokenize_words(clean_text(d)) for d in docs]
            tokenized = [[w for w in doc if len(w) > 2] for doc in tokenized]
            with st.spinner("Training LDA model..."):
                lda_model, corpus, dictionary = train_lda_model(tokenized, num_topics=num_topics, passes=20)
                topics = get_topics(lda_model, num_words=8)

            for t in topics:
                words_html = " ".join([f'<span class="tag">{w}</span>' for w in t["words"]])
                st.markdown(f'**Topic {t["topic_id"]+1}:** {words_html}', unsafe_allow_html=True)

# ==============================================================================
# PAGE: NER & Parsing
# ==============================================================================
elif page == "🏷️ NER & Parsing":
    st.markdown('<div class="section-header">🏷️ Named Entity Recognition & Dependency Parsing</div>', unsafe_allow_html=True)

    text_input = st.text_area("Enter text:",
        value="Apple Inc. was founded by Steve Jobs in Cupertino, California. The company launched the iPhone in January 2007 and it generated $200 billion in revenue.",
        height=100)

    tabs = st.tabs(["Named Entities", "Dependency Parsing", "POS Tags"])

    with tabs[0]:
        if st.button("🏷️ Extract Entities", type="primary"):
            entities = extract_entities(text_input)
            if entities:
                st.dataframe(pd.DataFrame(entities), use_container_width=True)
                html = render_ner_html(text_input)
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No named entities found.")

    with tabs[1]:
        if st.button("🌳 Parse Dependencies"):
            deps = parse_dependencies(text_input)
            st.dataframe(pd.DataFrame(deps), use_container_width=True)
            svg = render_dep_html(text_input)
            st.markdown(svg, unsafe_allow_html=True)

    with tabs[2]:
        if st.button("🔤 POS Tags"):
            tags = pos_tag(text_input)
            st.dataframe(pd.DataFrame(tags), use_container_width=True)

# ==============================================================================
# PAGE: Summarization
# ==============================================================================
elif page == "📝 Summarization":
    st.markdown('<div class="section-header">📝 Extractive Text Summarization</div>', unsafe_allow_html=True)

    text_input = st.text_area("Enter a long text to summarize:",
        value="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. "
              "The goal is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. "
              "NLP combines computational linguistics with statistical, machine learning, and deep learning models. "
              "These technologies enable computers to process human language in the form of text or voice data. "
              "Some of the most common NLP tasks include sentiment analysis, named entity recognition, machine translation, text summarization, and question answering. "
              "Recent advances in transformer models like BERT and GPT have significantly improved the state of the art in NLP. "
              "These models are pre-trained on large corpora of text and can be fine-tuned for specific downstream tasks. "
              "The applications of NLP are vast, ranging from virtual assistants and chatbots to automated content generation and medical text analysis.",
        height=200)

    num_sentences = st.slider("Summary Length (sentences)", 1, 10, 3)

    if st.button("📝 Summarize", type="primary"):
        result = extractive_summarize(text_input, num_sentences=num_sentences)
        st.markdown("**Summary:**")
        st.markdown(f'<div class="result-box">{result["summary"]}</div>', unsafe_allow_html=True)

        with st.expander("📊 Sentence Scores"):
            scores_df = pd.DataFrame([
                {"Sentence": s[:100]+"..." if len(s)>100 else s, "Score": sc}
                for s, sc in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(scores_df, use_container_width=True)

# ==============================================================================
# PAGE: BERT Prediction
# ==============================================================================
elif page == "🤖 BERT Prediction":
    st.markdown('<div class="section-header">🤖 BERT / DistilBERT Sentiment Classification</div>', unsafe_allow_html=True)
    st.markdown("Uses `distilbert-base-uncased-finetuned-sst-2-english` for sentiment prediction.")

    text_input = st.text_area("Enter text:",
        value="The new restaurant downtown has incredible food and the staff is very friendly!",
        height=100)

    if st.button("🤖 BERT Predict", type="primary"):
        with st.spinner("Running DistilBERT inference..."):
            result = bert_predict_sentiment(text_input)

        color = "tag-green" if result["label"] == "POSITIVE" else "tag-red"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Prediction</h3>
            <div class="value"><span class="tag {color}" style="font-size:1.2rem">{result['label']}</span></div>
            <p style="margin-top:1rem">Confidence: <b>{result['score']:.4f}</b> ({result['score']*100:.1f}%)</p>
            <p style="font-size:0.8rem;color:#888">Model: {result['model']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Batch BERT
    st.markdown("---")
    st.markdown("**Batch BERT Prediction**")
    batch = st.text_area("Multiple texts (one per line):", height=120, key="bert_batch",
        value="I absolutely loved this experience!\nThis was the worst service ever.\nThe weather is nice today.\nI'm extremely disappointed with the quality.")

    if st.button("🤖 Batch Predict"):
        texts = [t.strip() for t in batch.split("\n") if t.strip()]
        with st.spinner("Running batch inference..."):
            from src.nlp_tasks import bert_predict_batch
            results = bert_predict_batch(texts)
        rows = [{"Text": t[:80], "Label": r["label"], "Score": f'{r["score"]:.4f}'}
                for t, r in zip(texts, results)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.85rem'>"
    "🧠 TextAnalytica — CSI324 Text Analytics | Built with Streamlit, scikit-learn, spaCy & Transformers"
    "</p>", unsafe_allow_html=True
)
