import streamlit as st
import random
import requests
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
from datetime import datetime

# Imposta layout standard (non wide)
st.set_page_config(
    page_title="Themis AI",
    page_icon="logo.png",
    layout="centered"
)

# Custom CSS per migliorare l'aspetto - colori allineati con Fact Checker
st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #FF4081, #29B6F6, #00E676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}

.nav-button {
    background: linear-gradient(135deg, #66BB6A, #4CAF50);
    color: white;
    padding: 16px 32px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: bold;
    font-size: 16px;
    display: inline-block;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    text-align: center;
    min-width: 200px;
}

.nav-button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(102, 187, 106, 0.4);
    text-decoration: none;
    color: white;
}

.nav-button.fact-checker {
    background: linear-gradient(135deg, #29B6F6, #1976D2);
}

.nav-button.fact-checker:hover {
    box-shadow: 0 6px 20px rgba(41, 182, 246, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Centra l'immagine
col1, col2, col3 = st.columns([0.7, 1.15, 0.7])
with col2:
    st.image("logo.png", width=500)

# --- SEZIONE OVERVIEW ---
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 2px solid #66BB6A; border-radius: 15px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 8px 32px rgba(102, 187, 106, 0.1);">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <span style="font-size: 28px;">🎯</span>
            <h2 style="color: #66BB6A; margin: 0;">Overview</h2>
        </div>
        <div style="color: white; font-size: 16px; line-height: 1.6;">
            This project addresses the growing need for <b>fast and reliable verification</b> of online information. With the exponential growth of digital content and the proliferation of <b>misinformation</b>, there is an urgent demand for automated fact-checking tools that can assist users in distinguishing verified facts from false or misleading claims.
            <br><br>
            We leverage a comprehensive and continuously updated database of textual content sourced from multiple channels. This data is processed using <b>advanced Natural Language Processing (NLP) techniques and artificial intelligence</b> to create a robust verification system.
            <br><br>
            Our goal is to build a reliable tool that delivers <b>motivated and transparent assessments</b>, empowering users to make informed decisions about the information they encounter online.
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- SEZIONE DATASET ---
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 2px solid #FFB300; border-radius: 15px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 8px 32px rgba(255, 179, 0, 0.1);">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <span style="font-size: 28px;">📊</span>
            <h2 style="color: #FFB300; margin: 0;">Data</h2>
        </div>
        <div style="color: white; font-size: 16px; line-height: 1.6;">
            The dataset for this project is sourced from the <b>Webhose free news datasets</b>, publicly available on <a href="https://github.com/Webhose/free-news-datasets" target="_blank" style="color: #29B6F6;">GitHub</a>. The dataset contains comprehensive news <b>articles</b> that include not only the full text content but also rich <b>metadata</b> such as title, author, publication date, language, category, and much more.
            <br><br>
            All <b>articles</b> are stored in a <b>MongoDB database</b>. The database receives daily updates of approximately <b>10 new articles per day</b> through the <a href="https://webz.io/products/news-api#lite" target="_blank" style="color: #00E676;">Webz.io News API</a>, ensuring our dataset remains current and comprehensive with fresh content.
            <br><br>
            Users can explore detailed analytics of the original dataset through our <b>Data Analytics page</b>, which features an interactive dashboard directly connected to the MongoDB database. This dashboard delivers <b>comprehensive statistics and dynamic visualizations</b>, providing insights into data patterns, source distribution, and content trends.
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- SEZIONE METODOLOGIA ---
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 2px solid #AB47BC; border-radius: 15px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 8px 32px rgba(171, 71, 188, 0.1);">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <span style="font-size: 28px;">⚙️</span>
            <h2 style="color: #AB47BC; margin: 0;">Retrieval Augmented Generation</h2>
        </div>
        <div style="color: white; font-size: 16px; line-height: 1.6;">
            Our approach to fact-checking consists of a well-structured pipeline that combines advanced data preprocessing, semantic search, and large language models (LLMs) for intelligent reasoning and response generation.
            <br><br>
            <strong style="color: #00E676; font-size: 18px;">🔧 Data Preprocessing Pipeline</strong><br>
            The first stage involves comprehensive text processing to prepare articles for semantic analysis:<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Text Cleaning and Normalization: Standardizing format and removing noise from raw articles<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Named Entity Recognition (NER): Extracting key entities (people, organizations, and locations)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Content Summarization: Creating concise representations of lengthy articles<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Vector Embeddings Generation: Using paraphrase-multilingual-MiniLM-L12-v2 to create semantic embeddings for similarity search<br>
            All processed data and embeddings are stored in MongoDB for efficient retrieval and analysis.
            <br><br>
            <strong style="color: #29B6F6; font-size: 18px;">🔍 Semantic Search Engine</strong><br>
            When a user submits a claim for verification, our system performs intelligent information retrieval:<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Vector Similarity Search: Matching the claim against preprocessed article chunks using semantic embeddings<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Relevance Scoring: Ranking results based on semantic similarity scores<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>•</strong> Content Selection & Authority Filtering: Identifying the most relevant articles and passages from high-quality, authoritative sources for fact-checking
            <br><br>
            <strong style="color: #FF7043; font-size: 18px;">🤖 LLM-Powered Fact Verification</strong><br>
            The final stage leverages large language models for comprehensive analysis through Retrieval-Augmented Generation (RAG), combining selected evidence with advanced reasoning capabilities.
            <br>
            The system offers flexibility in model selection, utilizing either local Gemma 3B or Gemini 2.0 Flash via API for response generation.
            <br>
            Throughout the verification process, the system provides explainable reasoning with transparent analysis, culminating in a final credibility score that quantifies the claim's reliability based on the available evidence.
        </div>
    </div>
    """, unsafe_allow_html=True
)

# Divider grafico semplice
st.markdown(
    """
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <div style="height: 2px; background: #444; border-radius: 2px; margin: 0 auto; width: 60%;"></div>
    </div>
    """, unsafe_allow_html=True
)

# --- SEZIONE RANDOM ARTICLE ---
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin: 1.5rem 0;">
        <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 2px solid #F44336; border-radius: 15px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(244, 67, 54, 0.1); display: inline-block;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 32px;">📰</span>
                <h2 style="color: #F44336; margin: 0; font-size: 1.8rem; white-space: nowrap;">Random Article from Database</h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- Visualizzazione articolo casuale ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Estrae un articolo casuale con URL funzionante e immagine valida
def get_random_article_with_valid_url_and_image():
    count = collection.count_documents({})
    if count == 0:
        return None
    max_attempts = 15
    for _ in range(max_attempts):
        rand_idx = random.randint(0, count - 1)
        doc = collection.find().skip(rand_idx).limit(1)[0]
        url = doc.get('url', '')
        main_image = doc.get('main_image', '')
        
        # Verifica URL dell'articolo
        if url:
            try:
                resp = requests.head(url, timeout=3)
                if resp.status_code < 400:
                    # Se c'è un'immagine, verifica anche quella
                    if main_image and main_image.strip():
                        try:
                            img_resp = requests.head(main_image, timeout=3)
                            if img_resp.status_code < 400:
                                return doc
                            else:
                                continue  # Immagine non valida, prova altro articolo
                        except Exception:
                            continue  # Errore nell'accesso all'immagine, prova altro articolo
                    else:
                        return doc  # Articolo senza immagine ma con URL valido
            except Exception:
                continue
    return None

# Inizializza articolo casuale in session_state se non presente
if "random_article" not in st.session_state:
    st.session_state.random_article = get_random_article_with_valid_url_and_image()

# Funzione per aggiornare l'articolo casuale
def refresh_article():
    st.session_state.random_article = get_random_article_with_valid_url_and_image()

if st.session_state.random_article:
    article = st.session_state.random_article
    
    # Mostra immagine solo se esiste e non è vuota
    main_image = article.get('main_image', '')
    if main_image and main_image.strip():
        st.markdown(
            f"""
            <div style="position: relative; width: 100%; min-height: 220px; margin-bottom: 1.5rem;">
                <img src="{main_image}" style="width:100%; max-height:220px; object-fit:cover; border-radius: 12px; filter: brightness(0.6);" onerror="this.style.display='none'; this.parentElement.style.minHeight='auto'; this.parentElement.style.marginBottom='0';" />
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: flex-start; padding: 1.2rem;">
                    <span style="color: #FFD600; font-size: 1.3rem; font-weight: bold; text-shadow: 2px 2px 8px #000;">{article.get('title','')}</span>
                    <span style="color: #FFF; font-size: 1.1rem; text-shadow: 1px 1px 8px #000;">by {article.get('author','Unknown')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Se non c'è immagine, mostra solo titolo e autore
        st.markdown(
            f"""
            <div style="margin-bottom: 1.5rem; padding: 1.2rem; background: #333; border-radius: 12px;">
                <h3 style="color: #FFD600; font-size: 1.3rem; font-weight: bold; margin: 0 0 0.5rem 0;">{article.get('title','')}</h3>
                <span style="color: #FFF; font-size: 1.1rem;">by {article.get('author','Unknown')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Mostra il testo dell'articolo in un box con stile coerente
    st.markdown(
        f"""
        <div style="
            background: #222;
            color: #fff;
            border: 2px solid #29B6F6;
            border-radius: 12px;
            padding: 1.2rem;
            height: 220px;
            overflow-y: auto;
            font-size: 1.08rem;
            margin-bottom: 1.2rem;
            line-height: 1.6;
        ">
            {article.get('text','')}
        """,
        unsafe_allow_html=True
    )

    # Data di pubblicazione, link all'articolo e bottone refresh su stessa riga
    col1, col2, col3 = st.columns([1, 1, 1])
  
    with col1:
        pub_date = article.get('publish_date', 'Unknown')
        if isinstance(pub_date, datetime):
            pub_date = pub_date.strftime("%d %b %Y")
        st.markdown(f"<div style='text-align: center;'><strong>📅 Published:</strong> {pub_date}</div>", unsafe_allow_html=True)

    with col2:
        article_url = article.get('url', '')
        if article_url:
            st.link_button("🔗 Read Original Article", article_url)
    
    with col3:
        if st.button("🔄 New Random Article", key="refresh_article"):
            refresh_article()
            st.rerun()

# Chiudi il contenitore della sezione Random Article
st.markdown("</div>", unsafe_allow_html=True)

# --- SEZIONE CALL TO ACTION ---
st.markdown(
    """
    <div style="text-align: center; margin: 4rem 0 2rem 0;">
        <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 2px solid #444; border-radius: 15px; padding: 2rem;">
            <h3 style="color: #FFF; margin-bottom: 1.5rem;">🚀 Ready to Get Started?</h3>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <a href="/Data_Analytics" target="_self" style="background: linear-gradient(135deg, #FF4081, #E91E63); color: white; padding: 16px 32px; border-radius: 12px; text-decoration: none; font-weight: bold; font-size: 16px; display: inline-block; transition: all 0.3s ease; text-align: center; min-width: 200px;">
                    📊 Data Analytics
                </a>
                <a href="/Fact_Checker" target="_self" style="background: linear-gradient(135deg, #FF7043, #F4511E); color: white; padding: 16px 32px; border-radius: 12px; text-decoration: none; font-weight: bold; font-size: 16px; display: inline-block; transition: all 0.3s ease; text-align: center; min-width: 200px;">
                    🔍 Fact Checker
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

