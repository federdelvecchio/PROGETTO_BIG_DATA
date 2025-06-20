import streamlit as st          # Framework per applicazioni web interattive
import pandas as pd             # Libreria per manipolazione e analisi di dati tabulari
import plotly.express as px     # Libreria per creazione di grafici interattivi
from pymongo import MongoClient # Client per connessione e interazione con MongoDB
import pycountry               # Libreria per conversione codici paese ISO
from wordcloud import WordCloud # Libreria per generazione di word cloud
import matplotlib.pyplot as plt # Libreria per creazione di grafici statici
from config import MONGO_URI, DB_NAME, COLLECTION_NAME  # Configurazioni database
import numpy as np             # Libreria per operazioni numeriche avanzate

# Dimensione del batch per le query MongoDB - ottimizza il caricamento dei dati
# Un valore troppo alto può causare problemi di memoria, troppo basso rallenta le operazioni
BATCH_SIZE = 10000

# ================================
# FUNZIONI DI CONVERSIONE PAESI
# ================================

def iso2_to_country_name(iso2):
    """
    Converte i codici ISO2 dei paesi (es. 'IT') in nomi completi (es. 'Italy').
    Utilizza la libreria pycountry per la conversione standardizzata.
    In caso di errore restituisce il codice originale invariato.
    """
    try:
        return pycountry.countries.get(alpha_2=iso2).name
    except:
        return iso2

def country_name_to_iso2(country_name):
    """
    Converte i nomi completi dei paesi (es. 'Italy') in codici ISO2 (es. 'IT').
    Prima prova una conversione diretta, poi effettua una ricerca iterativa
    tra tutti i paesi disponibili in pycountry.
    """
    try:
        return pycountry.countries.get(name=country_name).alpha_2
    except:
        # Ricerca iterativa tra tutti i paesi se la conversione diretta fallisce
        for country in pycountry.countries:
            if country.name == country_name:
                return country.alpha_2
        return country_name

def iso2_to_iso3(iso2):
    """
    Converte i codici ISO2 (2 caratteri) in codici ISO3 (3 caratteri).
    I codici ISO3 sono necessari per alcune visualizzazioni geografiche
    come le mappe coropletiche di Plotly.
    """
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except:
        return None

# ================================
# FUNZIONI DI CARICAMENTO DATI
# ================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_base_data():
    """
    Carica i dati di base da MongoDB per popolare le opzioni dei filtri.
    Utilizza aggregation pipeline per ottenere tutti i valori unici
    dei campi utilizzati nei filtri (lingue, paesi, sentiment, ecc.).
    
    Cache: 3600 secondi (1 ora) - i dati base cambiano raramente
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Pipeline di aggregazione per estrarre valori unici e statistiche di base
    pipeline = [
        {
            # Raggruppa tutti i documenti e crea set di valori unici per ogni campo
            "$group": {
                "_id": None,  # Raggruppa tutto in un singolo documento
                "languages": {"$addToSet": "$language"},        # Set di tutte le lingue
                "countries": {"$addToSet": "$country"},         # Set di tutti i paesi
                "site_types": {"$addToSet": "$site_type"},      # Set di tutti i tipi di sito
                "sentiments": {"$addToSet": "$sentiment"},      # Set di tutti i sentiment
                "min_domain_rank": {"$min": "$domain_rank"},    # Valore minimo domain rank
                "max_domain_rank": {"$max": "$domain_rank"},    # Valore massimo domain rank
                "min_published": {"$min": "$published"},        # Data pubblicazione più antica
                "max_published": {"$max": "$published"}         # Data pubblicazione più recente
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    if result:
        data = result[0]
        
        # Filtra e ordina le lingue rimuovendo valori nulli
        languages = sorted([lang for lang in data.get("languages", []) if lang])
        
        # Conversione dei codici paese ISO2 in nomi completi per l'interfaccia utente
        countries_iso2 = [country for country in data.get("countries", []) if country]
        countries_full_names = []
        for iso2 in countries_iso2:
            full_name = iso2_to_country_name(iso2)
            countries_full_names.append(full_name)
        countries = sorted(list(set(countries_full_names)))  # Rimuove duplicati e ordina
        
        # Filtra e ordina altri campi rimuovendo valori nulli
        site_types = sorted([site_type for site_type in data.get("site_types", []) if site_type])
        sentiments = sorted([sentiment for sentiment in data.get("sentiments", []) if sentiment])
        
        return {
            "languages": languages,
            "countries": countries,
            "site_types": site_types,
            "sentiments": sentiments,
            # Costruisce la tupla del range domain rank con valori di default se mancanti
            "domain_rank_range": (
                int(data.get("min_domain_rank", 0)) if data.get("min_domain_rank") else 0,
                int(data.get("max_domain_rank", 1000000)) if data.get("max_domain_rank") else 1000000
            ),
            # Range delle date di pubblicazione
            "date_range": (
                data.get("min_published"),
                data.get("max_published")
            )
        }
    
    # Valori di default se la query non restituisce risultati
    return {
        "languages": [],
        "countries": [],
        "site_types": [],
        "sentiments": [],
        "domain_rank_range": (0, 1000000),
        "date_range": (None, None)
    }

@st.cache_data(show_spinner=False, ttl=7200)
def get_global_stats():
    """
    Ottiene le statistiche globali del database per le metriche principali.
    Calcola: numero totale articoli, dimensione database, lunghezza media testo.
    
    Cache: 7200 secondi (2 ore) - le statistiche globali cambiano lentamente
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Conta totale documenti (più veloce di count_documents per grandi collezioni)
    total_articles = collection.estimated_document_count()
    
    # Ottiene statistiche del database (dimensione, storage, ecc.)
    db_stats = db.command("dbstats")
    db_size_gb = db_stats["dataSize"] / (1024 * 1024 * 1024)  # Conversione in GB
    
    # Pipeline per calcolare la lunghezza media del testo degli articoli
    pipeline = [
        # Proietta solo il campo calcolato della lunghezza del testo
        {"$project": {"text_length": {"$strLenCP": "$text"}}},
        # Raggruppa tutto e calcola la media
        {"$group": {"_id": None, "avg_length": {"$avg": "$text_length"}}}
    ]
    
    result = list(collection.aggregate(pipeline))
    avg_text_len = result[0]["avg_length"] if result else 0
    
    return {
        "total_articles": total_articles,
        "db_size_gb": db_size_gb,
        "avg_text_len": avg_text_len
    }

@st.cache_data(show_spinner=False, ttl=1800)
def get_all_categories():
    """
    Ottiene tutte le categorie uniche dal database.
    Le categorie sono memorizzate come array, quindi usa $unwind per espanderle.
    
    Cache: 1800 secondi (30 minuti) - le categorie possono cambiare più frequentemente
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Pipeline per estrarre tutte le categorie uniche
    pipeline = [
        {"$unwind": "$categories"},                    # Espande l'array categories
        {"$group": {"_id": "$categories"}},           # Raggruppa per categoria unica
        {"$sort": {"_id": 1}}                         # Ordina alfabeticamente
    ]
    
    result = list(collection.aggregate(pipeline))
    # Filtra valori nulli e restituisce lista ordinata
    return sorted([item["_id"] for item in result if item["_id"]])

def build_filter_query(filters):
    """
    Costruisce la query MongoDB dai filtri applicati dall'utente.
    Converte i filtri dell'interfaccia utente in una query MongoDB valida.
    Gestisce diversi tipi di filtri: liste, range, date.
    """
    query = {}
    
    # Filtro per lingue: usa operatore $in per match multipli
    if filters.get("languages"):
        query["language"] = {"$in": filters["languages"]}
    
    # Filtro per paesi: converte nomi in codici ISO2 prima della query
    if filters.get("countries"):
        iso2_codes = [country_name_to_iso2(country) for country in filters["countries"]]
        query["country"] = {"$in": iso2_codes}
    
    # Filtro per tipi di sito
    if filters.get("site_types"):
        query["site_type"] = {"$in": filters["site_types"]}
    
    # Filtro per sentiment
    if filters.get("sentiments"):
        query["sentiment"] = {"$in": filters["sentiments"]}
    
    # Filtro per categorie: usa $in perché le categorie sono in array
    if filters.get("categories"):
        query["categories"] = {"$in": filters["categories"]}
    
    # Filtro per range domain rank: usa $gte (>=) e $lte (<=)
    if filters.get("domain_rank_range"):
        min_rank, max_rank = filters["domain_rank_range"]
        query["domain_rank"] = {"$gte": min_rank, "$lte": max_rank}
    
    # Filtro per range di date: gestione complessa per diversi formati di data
    if filters.get("date_range"):
        date_range = filters["date_range"]
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            if start_date and end_date:
                # Normalizza le date in formato date se sono datetime
                if hasattr(start_date, 'date'):
                    start_date = start_date.date()
                if hasattr(end_date, 'date'):
                    end_date = end_date.date()
                
                # Crea pattern per il matching delle date in formato ISO
                start_pattern = start_date.strftime("%Y-%m-%d")
                end_pattern = end_date.strftime("%Y-%m-%d")
                
                if start_date == end_date:
                    # Se le date sono uguali, usa regex per match del singolo giorno
                    query["published"] = {"$regex": f"^{start_pattern}"}
                else:
                    # Se sono diverse, usa range con fine giornata per end_date
                    query["published"] = {
                        "$gte": start_pattern,
                        "$lte": end_pattern + "T23:59:59.999Z"
                    }
    
    return query

@st.cache_data(show_spinner=False, ttl=900)
def load_filtered_data(_filters_key, filters):
    """
    Carica i dati filtrati da MongoDB con elaborazione in batch.
    Mostra una barra di progresso per operazioni lunghe.
    Proietta solo i campi necessari per ottimizzare le performance.
    
    Cache: 900 secondi (15 minuti) - i dati filtrati cambiano frequentemente
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Costruisce la query dai filtri e conta i documenti totali
    query = build_filter_query(filters)
    total_count = collection.count_documents(query)
    
    # Se non ci sono documenti, restituisce DataFrame vuoto
    if total_count == 0:
        return pd.DataFrame()
    
    # Inizializza elementi UI per il progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    docs = []
    processed = 0
    
    # Crea cursor con proiezione per limitare i campi scaricati
    # Questo riduce significativamente il traffico di rete e l'uso di memoria
    cursor = collection.find(query, {
        "language": 1, "country": 1, "sentiment": 1, "site_categories": 1,
        "updated": 1, "author": 1, "text": 1, "published": 1,
        "categories": 1, "site": 1, "domain_rank": 1, "title": 1,
        "site_type": 1, "main_image": 1
    }).batch_size(BATCH_SIZE)
    
    # Processa i documenti in batch per ottimizzare la memoria
    for doc in cursor:
        docs.append(doc)
        processed += 1
        
        # Aggiorna la barra di progresso ogni 1000 documenti
        if processed % 1000 == 0:
            progress = processed / total_count
            progress_bar.progress(progress)
            status_text.text(f"{processed:,}/{total_count:,} documents")
    
    # Finalizza la barra di progresso e pulisce l'interfaccia
    progress_bar.progress(1.0)
    status_text.text(f"Loaded {processed:,} documents")
    progress_bar.empty()
    status_text.empty()
    
    # Converte la lista di documenti in DataFrame pandas
    return pd.DataFrame(docs)

# ================================
# FUNZIONI DI ANALISI
# ================================

@st.cache_data(show_spinner=False, ttl=1800)
def get_null_percentages_filtered(_filters_key, filters):
    """
    Calcola la percentuale di valori null per ogni campo nei dati filtrati.
    Analizza sia i campi semplici che quelli nested (entità).
    Restituisce un DataFrame con field name e percentuale di valori mancanti.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Applica i filtri e conta il totale
    query = build_filter_query(filters)
    total = collection.count_documents(query)
    
    if total == 0:
        return pd.DataFrame()
    
    # Lista di tutti i campi da analizzare, inclusi quelli nested
    fields = [
        "uuid", "url", "author", "published", "title", "text", "language", "sentiment",
        "categories", "crawled", "updated", "site", "site_type", "country", "main_image",
        "domain_rank", "domain_rank_updated", "site_categories",
        "entities.persons.name", "entities.persons.sentiment",
        "entities.organizations.name", "entities.organizations.sentiment",
        "entities.locations.name", "entities.locations.sentiment"
    ]
    
    results = []
    for field in fields:
        # Crea query per contare documenti dove il campo è mancante o null
        field_query = query.copy()
        field_query.update({
            "$or": [
                {field: {"$exists": False}},  # Il campo non esiste
                {field: None}                 # Il campo esiste ma è null
            ]
        })
        
        missing = collection.count_documents(field_query)
        percent = (missing / total * 100) if total > 0 else 0
        results.append({"Field": field, "Missing (%)": f"{percent:.2f}"})
    
    return pd.DataFrame(results)

@st.cache_data(show_spinner=False, ttl=1800)
def get_top_entities_filtered(_filters_key, filters, entity_type, limit=5):
    """
    Recupera le entità più frequenti dai dati MongoDB filtrati.
    Gestisce tre tipi di entità: persons, organizations, locations.
    Usa aggregation pipeline per contare le occorrenze di ogni entità.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    query = build_filter_query(filters)
    
    # Pipeline di aggregazione per estrarre le top entità
    pipeline = [
        {"$match": query},                                              # Applica filtri
        {"$unwind": f"$entities.{entity_type}"},                      # Espande l'array delle entità
        {"$group": {"_id": f"$entities.{entity_type}.name", "count": {"$sum": 1}}},  # Conta occorrenze
        {"$sort": {"count": -1}},                                      # Ordina per count decrescente
        {"$limit": limit}                                              # Limita ai top N risultati
    ]
    
    result = list(collection.aggregate(pipeline))
    df = pd.DataFrame(result)
    
    # Rinomina le colonne per una presentazione migliore
    if not df.empty:
        df.rename(columns={'_id': 'Entity', 'count': 'Count'}, inplace=True)
    return df

# ================================
# FUNZIONI DI CREAZIONE GRAFICI
# ================================

@st.cache_data(show_spinner=False, ttl=1800)
def create_language_chart(_filters_key, df):
    """
    Crea un grafico a barre per la distribuzione delle lingue.
    Utilizza una palette di colori vivaci e gestisce i valori mancanti.
    """
    if df.empty or 'language' not in df.columns:
        return None
    
    # Conta le occorrenze di ogni lingua, sostituendo null con "Unknown"
    lang_counts = df['language'].fillna("Unknown").value_counts().sort_values(ascending=False)
    if lang_counts.empty:
        return None
        
    # Palette di colori vivaci per distinguere meglio le lingue
    bright_palette = ["#FFB300", "#FF7043", "#29B6F6", "#AB47BC", "#66BB6A", "#FFD600", "#FF4081", "#00E676"]
    
    # Crea grafico a barre con Plotly Express
    fig_lang = px.bar(x=lang_counts.index, y=lang_counts.values, 
                      labels={'x': 'Language', 'y': 'Number of articles'}, 
                      color=lang_counts.index, color_discrete_sequence=bright_palette)
    
    # Personalizza il layout per il tema scuro di Streamlit
    fig_lang.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=40, b=10), 
                          font_color="white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig_lang

@st.cache_data(show_spinner=False, ttl=1800)
def create_geographic_chart(_filters_key, df):
    """
    Crea una mappa coropletica per la distribuzione geografica degli articoli.
    Converte i codici ISO2 in ISO3 necessari per la visualizzazione Plotly.
    """
    if df.empty or 'country' not in df.columns:
        return None
    
    # Conta articoli per paese, gestendo valori null e spazi vuoti
    country_counts = df['country'].replace({None: "Unknown", " ": "Unknown"}).value_counts()
    country_counts = country_counts.drop("Unknown", errors='ignore')  # Rimuove "Unknown" se presente
    
    if country_counts.empty:
        return None
    
    # Prepara DataFrame per la mappa con conversioni di codici paese
    country_df = country_counts.reset_index()
    country_df.columns = ['country_iso2', 'count']
    country_df['country_name'] = country_df['country_iso2'].apply(iso2_to_country_name)
    country_df['iso3'] = country_df['country_iso2'].apply(iso2_to_iso3)
    country_df = country_df.dropna(subset=['iso3'])  # Rimuove paesi senza codice ISO3 valido
    
    if country_df.empty:
        return None
        
    # Crea mappa coropletica con scala di colori personalizzata
    fig_map = px.choropleth(country_df, locations='iso3', locationmode='ISO-3', color='count', 
                           hover_name='country_name',
                           color_continuous_scale=["#FFF176", "#FFD54F", "#FFB300", "#FF7043", "#FF4081", "#AB47BC", "#29B6F6"], 
                           labels={'country_name': 'Country', 'count': 'Number of articles'})
    
    # Personalizza la geografia e il layout della mappa
    fig_map.update_geos(projection_type="natural earth", showcountries=True, countrycolor="white", 
                       showcoastlines=True, coastlinecolor="white", showland=True, 
                       landcolor="rgba(0,0,0,0)", bgcolor='rgba(0,0,0,0)')
    fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                         margin={"r":0,"t":40,"l":0,"b":0}, height=350, font_color="white")
    return fig_map

@st.cache_data(show_spinner=False, ttl=1800)
def create_sentiment_chart(_filters_key, df):
    """
    Crea un grafico a ciambella per l'analisi del sentiment.
    Usa colori semantici: verde per positivo, rosso per negativo, arancione per sconosciuto.
    """
    if df.empty or 'sentiment' not in df.columns:
        return None
    
    # Mappatura per normalizzare i valori di sentiment
    sentiment_map = {None: "Unknown", "": "Unknown"}
    df_copy = df.copy()
    df_copy['sentiment'] = df_copy['sentiment'].replace(sentiment_map)
    df_copy['sentiment'] = df_copy['sentiment'].fillna("Unknown").astype(str).str.lower()
    sent_counts = df_copy['sentiment'].value_counts()
    
    if sent_counts.empty:
        return None
        
    # Mappa colori semantici per i sentiment
    color_map = {"positive": "#00FF7F", "negative": "#FF355E", "unknown": "#FFB300"}
    colors = [color_map.get(label, "#CCCCCC") for label in sent_counts.index]
    
    # Crea grafico a torta con buco al centro (donut chart)
    fig = px.pie(names=sent_counts.index, values=sent_counts.values, 
                color_discrete_sequence=colors, hole=0.5)
    fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=30, b=30), height=350, 
                     font_color="white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

@st.cache_data(show_spinner=False, ttl=1800)
def create_timeline_chart(_filters_key, df):
    """
    Crea un grafico temporale delle pubblicazioni mensili.
    Converte le date in formato datetime e raggruppa per mese.
    """
    if df.empty or 'published' not in df.columns:
        return None
    
    df_copy = df.copy()
    # Conversione robusta delle date con gestione errori
    df_copy['published_dt'] = pd.to_datetime(df_copy['published'], errors='coerce', utc=True)
    df_copy['published_dt'] = df_copy['published_dt'].dt.tz_convert(None)  # Rimuove timezone
    df_time = df_copy[df_copy['published_dt'].notnull()].copy()  # Filtra date valide
    
    if df_time.empty:
        return None
        
    # Raggruppa per fine mese e conta gli articoli
    monthly_counts = (df_time.set_index('published_dt')
                     .resample('ME').size().rename("Articles").reset_index())
    
    # Crea grafico lineare con marcatori per ogni punto
    fig_time = px.line(monthly_counts, x='published_dt', y='Articles', 
                      labels={'published_dt': 'Month', 'Articles': 'Number of articles'}, 
                      markers=True)
    
    # Personalizza layout per tema scuro senza griglia
    fig_time.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), 
                          font_color="white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          xaxis=dict(tickformat="%b %Y", showgrid=False), yaxis=dict(showgrid=False))
    return fig_time

@st.cache_data(show_spinner=False, ttl=1800)
def create_wordcloud(_filters_key, df):
    """
    Crea una word cloud dalle categorie degli articoli.
    Gestisce categorie sia in formato lista che stringa.
    """
    if df.empty or 'categories' not in df.columns:
        return None
    
    all_categories = []
    # Estrae tutte le categorie gestendo diversi formati di dati
    for cats in df['categories'].dropna():
        if isinstance(cats, list):
            # Se è una lista, estende all_categories con elementi string
            all_categories.extend([str(cat) for cat in cats if isinstance(cat, str)])
        elif isinstance(cats, str):
            # Se è una stringa singola, la aggiunge direttamente
            all_categories.append(cats)

    # Unisce tutte le categorie in un unico testo
    categories_text = " ".join(all_categories)

    if not categories_text.strip():
        return None
        
    # Crea word cloud con configurazione personalizzata
    wordcloud = WordCloud(width=1200, height=500, background_color=None, mode="RGBA", 
                         colormap='plasma', max_words=120, prefer_horizontal=0.9, 
                         font_path=None, relative_scaling=0.5, min_font_size=10).generate(categories_text)
    
    # Crea figura matplotlib con sfondo trasparente
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # Nasconde gli assi
    fig.patch.set_alpha(0.0)  # Sfondo trasparente
    return fig

@st.cache_data(show_spinner=False, ttl=1800)
def create_domain_rank_chart(_filters_key, df):
    """
    Crea un istogramma per la distribuzione del ranking dei domini.
    Calcola e restituisce anche statistiche descrittive del dataset.
    """
    if df.empty or 'domain_rank' not in df.columns:
        return None, None
    
    # Converte in numerico e filtra valori validi (> 0)
    domain_ranks = pd.to_numeric(df['domain_rank'], errors='coerce')
    domain_ranks = domain_ranks[domain_ranks > 0].dropna()
    
    if domain_ranks.empty:
        return None, None
        
    # Calcola statistiche descrittive
    percentile_25 = domain_ranks.quantile(0.25)
    percentile_75 = domain_ranks.quantile(0.75)
    
    # Crea testo con statistiche formattate
    stats_text = f"**Range:** {int(domain_ranks.min())} - {int(domain_ranks.max())} &nbsp; | &nbsp; " \
                 f"**Mean:** {int(domain_ranks.mean())} &nbsp; | &nbsp; " \
                 f"**Median:** {int(domain_ranks.median())} &nbsp; | &nbsp; " \
                 f"**25th percentile:** {int(percentile_25)} &nbsp; | &nbsp; " \
                 f"**75th percentile:** {int(percentile_75)}"
    
    # Crea istogramma con 50 bin
    fig_rank_lin = px.histogram(domain_ranks, nbins=50, 
                               labels={'value': 'Domain Rank', 'count': 'Number of sites'}, 
                               color_discrete_sequence=["#FFD600"])
    
    # Personalizza layout per tema scuro
    fig_rank_lin.update_layout(height=350, font_color="white", plot_bgcolor='rgba(0,0,0,0)', 
                              paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=40, b=10), 
                              showlegend=False)
    return fig_rank_lin, stats_text

@st.cache_data(show_spinner=False, ttl=1800)
def create_site_categories_chart(_filters_key, df):
    """
    Crea un grafico a torta per le categorie dei siti web.
    Mostra le top 10 categorie più frequenti.
    """
    if df.empty or 'site_categories' not in df.columns:
        return None
    
    site_categories_list = []
    # Estrae tutte le categorie di sito gestendo diversi formati
    for cats in df['site_categories'].dropna():
        if isinstance(cats, list):
            # Se è lista, estrae elementi string validi e li pulisce
            site_categories_list.extend([str(cat).strip() for cat in cats if isinstance(cat, str) and str(cat).strip()])
        elif isinstance(cats, str) and cats.strip():
            # Se è stringa non vuota, la aggiunge dopo pulizia
            site_categories_list.append(cats.strip())

    if not site_categories_list:
        return None
        
    # Conta occorrenze e prende le top 10
    site_cat_counts = pd.Series(site_categories_list).value_counts().nlargest(10)
    if site_cat_counts.empty:
        return None
        
    # Palette colori vivaci per distinguere le categorie
    bright_palette = ["#FFB300", "#FF7043", "#29B6F6", "#AB47BC", "#66BB6A", "#FFD600", "#FF4081", "#00E676"]
    
    # Crea grafico a torta
    fig_site_pie = px.pie(names=site_cat_counts.index, values=site_cat_counts.values, 
                         color_discrete_sequence=bright_palette)
    
    # Personalizza layout con legenda posizionata a destra
    fig_site_pie.update_layout(height=400, font_color="white", plot_bgcolor='rgba(0,0,0,0)', 
                              paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=40, b=10), 
                              showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5))
    
    # Aggiunge bordo nero alle fette per migliore definizione
    fig_site_pie.update_traces(marker=dict(line=dict(color='#000000', width=1)))
    return fig_site_pie

def create_freshness_chart(df):
    """
    Crea un grafico per l'analisi della freschezza dei contenuti.
    Analizza il tempo trascorso tra pubblicazione e ultimo aggiornamento.
    Categorizza i ritardi in fasce temporali significative.
    """
    if df.empty or 'updated' not in df.columns or 'published' not in df.columns:
        return None, None
    
    try:
        # Conversione robusta delle date con gestione timezone
        df['updated_dt'] = pd.to_datetime(df['updated'], errors='coerce', utc=True)
        df['published_dt'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
        df['updated_dt'] = df['updated_dt'].dt.tz_convert(None)
        df['published_dt'] = df['published_dt'].dt.tz_convert(None)
        
        # Filtra solo record con entrambe le date valide
        mask = df['updated_dt'].notna() & df['published_dt'].notna()
        
        if mask.any():
            # Calcola differenza temporale in secondi
            time_diff = (df.loc[mask, 'updated_dt'] - df.loc[mask, 'published_dt'])
            seconds_diff = time_diff.dt.total_seconds()
            freshness_data = seconds_diff[seconds_diff >= 0]  # Solo differenze positive
            
            if not freshness_data.empty:
                # Categorizza i ritardi in fasce temporali significative
                categories = []
                for val in freshness_data:
                    if val == 0:
                        categories.append("Simultaneous")        # Aggiornamento simultaneo
                    elif val < 3600:
                        categories.append("0-1 hour")           # Entro un'ora
                    elif val < 43200:
                        categories.append("1-12 hours")         # Entro 12 ore
                    elif val < 86400:
                        categories.append("12-24 hours")        # Entro 24 ore
                    elif val < 604800:
                        categories.append("1 day - 1 week")     # Entro una settimana
                    else:
                        categories.append("1 week - 1 month")   # Oltre una settimana

                # Conta occorrenze per categoria
                cat_counts = pd.Series(categories).value_counts()
                
                # Ordina le categorie in modo logico temporale
                order = ["Simultaneous", "0-1 hour", "1-12 hours", "12-24 hours", "1 day - 1 week", "1 week - 1 month"]
                cat_counts = cat_counts.reindex([cat for cat in order if cat in cat_counts.index])
                
                # Crea grafico a barre
                fig_fresh = px.bar(x=cat_counts.index, y=cat_counts.values, 
                                  labels={'x': 'Time delay category', 'y': 'Number of articles'}, 
                                  color_discrete_sequence=["#29B6F6"])
                
                # Personalizza layout con etichette inclinate
                fig_fresh.update_layout(height=350, font_color="white", plot_bgcolor='rgba(0,0,0,0)', 
                                       paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=40, b=10), 
                                       xaxis_tickangle=-45)
                
                # Calcola metriche statistiche per i widget
                same_time = (freshness_data == 0).sum()          # Aggiornamenti simultanei
                avg_seconds = freshness_data.mean()              # Ritardo medio
                max_seconds = freshness_data.max()               # Ritardo massimo
                mode_seconds = freshness_data.mode()             # Moda (valore più frequente)
                mode_val = mode_seconds.iloc[0] if not mode_seconds.empty else 0
                
                metrics = {
                    'same_time': same_time,
                    'avg_seconds': avg_seconds,
                    'max_seconds': max_seconds,
                    'mode_val': mode_val
                }
                
                return fig_fresh, metrics
    except Exception:
        # In caso di errore nella elaborazione, restituisce None
        return None, None
    
    return None, None