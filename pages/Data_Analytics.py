import streamlit as st          # Framework per creare applicazioni web interattive in Python
import pandas as pd             # Libreria per manipolazione e analisi di dati strutturati
from pymongo import MongoClient # Client per connettersi e interagire con database MongoDB
from config import MONGO_URI, DB_NAME, COLLECTION_NAME  # Importa configurazioni database

# Importazione di tutte le funzioni di utilità per l'analisi dei dati
# Queste funzioni sono definite in un modulo separato per mantenere il codice organizzato
from utils.data_analytics_utils import (
    load_base_data,                    # Carica i dati di base per inizializzare l'applicazione
    get_global_stats,                  # Ottiene statistiche globali del dataset
    get_all_categories,                # Recupera tutte le categorie disponibili nel database
    load_filtered_data,                # Carica dati applicando i filtri selezionati dall'utente
    build_filter_query,               # Costruisce query MongoDB per filtrare i dati
    get_null_percentages_filtered,     # Calcola percentuali di valori mancanti nei dati filtrati
    get_top_entities_filtered,         # Estrae le entità più frequenti (persone, organizzazioni, luoghi)
    create_language_chart,             # Crea grafico distribuzione lingue
    create_geographic_chart,           # Crea mappa geografica della distribuzione articoli
    create_sentiment_chart,            # Crea grafico analisi sentiment
    create_timeline_chart,             # Crea timeline delle pubblicazioni nel tempo
    create_wordcloud,                  # Genera word cloud dalle categorie degli articoli
    create_domain_rank_chart,          # Crea grafico ranking popolarità domini
    create_site_categories_chart,      # Crea grafico categorie dei siti web
    create_freshness_chart             # Crea grafico analisi freschezza contenuti
)

# ================================
# INIZIALIZZAZIONE STATO SESSIONE
# ================================

# Streamlit utilizza st.session_state per mantenere i dati tra le varie esecuzioni della pagina
# Questo è fondamentale per preservare i filtri e i dati caricati quando l'utente interagisce con l'app

# Inizializza il dizionario dei filtri se non esiste già
# I filtri vengono mantenuti in sessione per evitare di perderli ad ogni ricaricamento
if 'filters' not in st.session_state:
    st.session_state.filters = {}

# Carica i dati di base una sola volta per sessione per ottimizzare le performance
# Questi dati includono le opzioni disponibili per tutti i filtri (lingue, paesi, ecc.)
if 'base_data' not in st.session_state:
    with st.spinner("Loading base data..."):  # Mostra spinner durante il caricamento
        st.session_state.base_data = load_base_data()

# Flag per gestire il primo caricamento della pagina
# Imposta valori di default per tutti i filtri al primo accesso
if 'first_load_done' not in st.session_state:
    st.session_state.first_load_done = True
    # Inizializza tutti i filtri con valori di default
    st.session_state.filters = {
        "languages": [],        # Lista vuota = nessun filtro su lingue
        "countries": [],        # Lista vuota = nessun filtro su paesi
        "site_types": [],       # Lista vuota = nessun filtro su tipi di sito
        "sentiments": [],       # Lista vuota = nessun filtro su sentiment
        "categories": [],       # Lista vuota = nessun filtro su categorie
        "domain_rank_range": st.session_state.base_data["domain_rank_range"],  # Range completo
        "date_range": None      # Nessun filtro su date
    }

# ================================
# INTERFACCIA UTENTE PRINCIPALE
# ================================

# Titolo principale della pagina di analisi dati
st.title("Data Analytics")

# Sezione filtri racchiusa in un expander per ottimizzare lo spazio
# expanded=False significa che inizialmente è chiusa
with st.expander("🔍 **Data Filters**", expanded=False):
    # Utilizzo di st.form per raggruppare tutti i controlli di filtro
    # Questo evita che la pagina si ricarichi ad ogni modifica di un singolo filtro
    with st.form("filters_form", border=False):
        # Divisione dei filtri in due colonne per un layout più organizzato
        col1, col2 = st.columns(2)
        
        # COLONNA SINISTRA - Filtri base
        with col1:
            # Multiselect per filtrare per lingue degli articoli
            # Mostra tutte le lingue disponibili e mantiene la selezione precedente
            selected_languages = st.multiselect(
                "Languages",
                options=st.session_state.base_data["languages"],
                default=st.session_state.filters.get("languages", [])
            )
            
            # Multiselect per filtrare per paesi di provenienza degli articoli
            selected_countries = st.multiselect(
                "Countries",
                options=st.session_state.base_data["countries"],
                default=st.session_state.filters.get("countries", [])
            )
            
            # Multiselect per filtrare per tipologia di sito web
            selected_site_types = st.multiselect(
                "Site Types",
                options=st.session_state.base_data["site_types"],
                default=st.session_state.filters.get("site_types", [])
            )
            
            # Multiselect per filtrare per sentiment degli articoli
            selected_sentiments = st.multiselect(
                "Sentiment",
                options=st.session_state.base_data["sentiments"],
                default=st.session_state.filters.get("sentiments", [])
            )

        # COLONNA DESTRA - Filtri avanzati
        with col2:
            # Caricamento categorie solo se non già presente in sessione
            # Le categorie sono molte quindi vengono caricate separatamente per ottimizzare
            if 'all_categories' not in st.session_state:
                with st.spinner("Loading categories..."):
                    st.session_state.all_categories = get_all_categories()
            
            # Multiselect per categorie degli articoli
            selected_categories = st.multiselect(
                "Categories",
                options=st.session_state.all_categories,
                default=st.session_state.filters.get("categories", [])
            )
            
            # Slider per filtrare per range di ranking del dominio
            # Domain rank: numero più basso = sito più popolare
            domain_range = st.session_state.base_data["domain_rank_range"]
            selected_domain_range = st.slider(
                "Domain Rank Range",
                min_value=domain_range[0],
                max_value=domain_range[1],
                value=st.session_state.filters.get("domain_rank_range", domain_range),
                step=1000  # Step di 1000 per rendere più fluida la selezione
            )

            # Controlli per filtro data: due date picker separati per start e end
            date_range = st.session_state.base_data["date_range"]
            if date_range[0] and date_range[1]:  # Verifica che esistano date valide
                # Conversione delle date string in oggetti date di pandas
                min_date = pd.to_datetime(date_range[0]).date()
                max_date = pd.to_datetime(date_range[1]).date()
                
                # Due colonne per start e end date
                col_start, col_end = st.columns(2)
                
                with col_start:
                    # Date picker per data di inizio
                    start_date = st.date_input(
                        "Start Date",
                        value=st.session_state.filters.get("date_range", (min_date, max_date))[0] if st.session_state.filters.get("date_range") else min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="start_date"
                    )
                
                with col_end:
                    # Date picker per data di fine
                    end_date = st.date_input(
                        "End Date",
                        value=st.session_state.filters.get("date_range", (min_date, max_date))[1] if st.session_state.filters.get("date_range") else max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="end_date"
                    )
                
                # Crea tupla del range date selezionato
                selected_date_range = (start_date, end_date) if start_date and end_date else None
            else:
                # Se non ci sono date disponibili nel dataset
                selected_date_range = None
                
            st.write("")  # Spazio vuoto per allineamento layout
            
            # Pulsante per applicare i filtri selezionati
            # type="primary" lo rende visivamente più prominente
            apply_filters = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)
    
    # Logica di applicazione dei filtri quando il pulsante viene premuto
    if apply_filters:
        # Processamento del range di date per assicurarsi che sia nel formato corretto
        processed_date_range = None
        if selected_date_range:
            # Varie verifiche per garantire che il range sia una tupla valida
            if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
                processed_date_range = selected_date_range
            elif hasattr(selected_date_range, '__len__') and len(selected_date_range) == 2:
                processed_date_range = tuple(selected_date_range)
        
        # Aggiornamento dello stato di sessione con i nuovi filtri
        st.session_state.filters = {
            "languages": selected_languages,
            "countries": selected_countries,
            "site_types": selected_site_types,
            "sentiments": selected_sentiments,
            "categories": selected_categories,
            "domain_rank_range": selected_domain_range,
            "date_range": processed_date_range
        }
        # Ricarica la pagina per applicare i nuovi filtri
        st.rerun()

# ================================
# CARICAMENTO E ANALISI DATI
# ================================

# Crea una chiave univoca basata sui filtri per il caching dei dati
# Questo evita di ricaricare gli stessi dati se i filtri non sono cambiati
filters_key = str(st.session_state.filters)

# Caricamento dei dati filtrati con indicatore di progresso
with st.spinner("Loading data..."):
    df = load_filtered_data(filters_key, st.session_state.filters)

# Verifica se esistono dati con i filtri applicati
if df.empty:
    st.warning("No data found with the current filters. Please adjust your filter criteria.")
    st.stop()  # Interrompe l'esecuzione se non ci sono dati

# ================================
# SEZIONE ANALISI E VISUALIZZAZIONI
# ================================

# Spazio verticale per migliorare il layout visivo
st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True)

# Caricamento delle statistiche globali del database
global_stats = get_global_stats()

# Connessione diretta a MongoDB per contare gli articoli filtrati
# Questo è più efficiente che contare dal DataFrame caricato
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
query = build_filter_query(st.session_state.filters)
num_articles = collection.count_documents(query)

# Visualizzazione delle metriche principali in 4 colonne
col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

with col_stats1:
    # Metrica: numero totale di articoli nel database
    st.markdown(f"""<div style="background: #222; border-radius: 12px; padding: 1.2rem 0.5rem; color: #66BB6A; font-size: 1.2rem; text-align:center;"><b>Total articles</b><br>{global_stats['total_articles']:,}</div>""", unsafe_allow_html=True)

with col_stats2:
    # Metrica: numero di articoli filtrati con percentuale
    percentage = (num_articles / global_stats['total_articles'] * 100) if global_stats['total_articles'] > 0 else 0
    st.markdown(f"""<div style="background: #222; border-radius: 12px; padding: 1.2rem 0.5rem; color: #FFD600; font-size: 1.2rem; text-align:center;"><b>Filtered articles</b><br>{num_articles:,} ({percentage:.1f}%)</div>""", unsafe_allow_html=True)

with col_stats3:
    # Metrica: dimensione del database in GB
    st.markdown(f"""<div style="background: #222; border-radius: 12px; padding: 1.2rem 0.5rem; color: #29B6F6; font-size: 1.2rem; text-align:center;"><b>Database size</b><br>{global_stats['db_size_gb']:.2f} GB</div>""", unsafe_allow_html=True)

with col_stats4:
    # Metrica: lunghezza media del testo degli articoli
    st.markdown(f"""<div style="background: #222; border-radius: 12px; padding: 1.2rem 0.5rem; color: #FF7043; font-size: 1.2rem; text-align:center;"><b>Avg. text length</b><br>{global_stats['avg_text_len']:.0f} chars</div>""", unsafe_allow_html=True)

st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True)

# ANALISI VALORI MANCANTI
st.subheader("📊 Missing values per field")
# Ottiene e visualizza la percentuale di valori mancanti per ogni campo
df_nulls = get_null_percentages_filtered(filters_key, st.session_state.filters)
if not df_nulls.empty:
    with st.container():
        # Visualizza tabella con percentuali di valori mancanti
        st.dataframe(df_nulls, use_container_width=True, height=300, hide_index=True)
    st.caption("Percentage of missing/null values for each field in filtered data.")
else:
    st.info("No data available for missing values analysis.")
st.divider()

# ANALISI DISTRIBUZIONE LINGUE
st.subheader("🌍 Language breakdown")
# Crea e visualizza grafico delle lingue più frequenti
fig_lang = create_language_chart(filters_key, df)
if fig_lang:
    st.plotly_chart(fig_lang, use_container_width=True)
    st.caption("Most frequent languages in the filtered dataset.")
else:
    st.info("No language data available.")
st.divider()

# ANALISI DISTRIBUZIONE GEOGRAFICA
st.subheader("🗺️ Geographical distribution")
# Crea mappa coropletica della distribuzione geografica degli articoli
fig_map = create_geographic_chart(filters_key, df)
if fig_map:
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Countries with more selected articles are shown with brighter colors.")
else:
    st.info("No geographical data available.")
st.divider()

# ANALISI SENTIMENT
st.subheader("😊 Sentiment scores")
# Crea grafico a ciambella per la distribuzione dei sentiment
fig_sentiment = create_sentiment_chart(filters_key, df)
if fig_sentiment:
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.caption("Donut chart distribution of sentiment labels in the filtered data.")
else:
    st.info("No sentiment data available.")
st.divider()

# TIMELINE PUBBLICAZIONI
st.subheader("📅 Monthly publications")
# Crea grafico temporale delle pubblicazioni mensili
fig_timeline = create_timeline_chart(filters_key, df)
if fig_timeline:
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.caption("Monthly trend of published articles in filtered data.")
else:
    st.info("No publication date available for articles.")
st.divider()

# WORD CLOUD CATEGORIE
st.subheader("🏷️ Article categories")
# Genera word cloud dalle categorie degli articoli
fig_wordcloud = create_wordcloud(filters_key, df)
if fig_wordcloud:
    st.pyplot(fig_wordcloud)  # pyplot per matplotlib figures
    st.caption("Word cloud generated from article categories in filtered data.")
else:
    st.info("No category data available.")
st.divider()

# ANALISI RANKING DOMINI
st.subheader("🏆 Website popularity")
# Crea istogramma della distribuzione del domain rank
fig_domain, stats_text = create_domain_rank_chart(filters_key, df)
if fig_domain and stats_text:
    st.write(stats_text)  # Visualizza statistiche testuali
    st.plotly_chart(fig_domain, use_container_width=True)
    st.caption("Lower domain rank = more popular site. Distribution is shown for filtered data.")
else:
    st.info("No domain_rank data available.")
st.divider()

# ANALISI FRESCHEZZA CONTENUTI
st.subheader("🔄 Content Freshness Analysis")
# Analizza il tempo trascorso tra pubblicazione e ultimo aggiornamento
fig_fresh, metrics = create_freshness_chart(df)
if fig_fresh and metrics:
    # Visualizza metriche di freschezza in 4 colonne
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Numero di articoli aggiornati contemporaneamente alla pubblicazione
        st.metric("Simultaneous updates", f"{metrics['same_time']:,}")
    
    with col2:
        # Ritardo medio tra pubblicazione e aggiornamento
        avg_seconds = metrics['avg_seconds']
        if avg_seconds < 3600:
            st.metric("Average delay", f"{avg_seconds/60:.1f} min")
        elif avg_seconds < 86400:
            st.metric("Average delay", f"{avg_seconds/3600:.1f} hrs")
        else:
            st.metric("Average delay", f"{avg_seconds/86400:.1f} days")
    
    with col3:
        # Valore più frequente (moda) del ritardo
        mode_val = metrics['mode_val']
        if mode_val == 0:
            st.metric("Mode", "0 hrs")
        elif mode_val < 3600:
            st.metric("Mode", f"{mode_val/60:.1f} min")
        elif mode_val < 86400:
            st.metric("Mode", f"{mode_val/3600:.1f} hrs")
        else:
            st.metric("Mode", f"{mode_val/86400:.1f} days")
    
    with col4:
        # Ritardo massimo osservato
        max_seconds = metrics['max_seconds']
        if max_seconds < 86400:
            st.metric("Max delay", f"{max_seconds/3600:.1f} hrs")
        else:
            st.metric("Max delay", f"{max_seconds/86400:.1f} days")

    st.plotly_chart(fig_fresh, use_container_width=True)
    st.caption("Time delay categories between article publication and last update in filtered data.")
else:
    st.info("'updated' field not available. Content freshness analysis requires both 'updated' and 'published' fields.")
st.divider()

# CATEGORIE SITI WEB
st.subheader("🌐 Top 10 site categories")
# Crea grafico a torta delle categorie di siti web più frequenti
fig_site_categories = create_site_categories_chart(filters_key, df)
if fig_site_categories:
    st.plotly_chart(fig_site_categories, use_container_width=True)
    st.caption("Pie chart showing the top 10 site categories in filtered data.")
else:
    st.info("Site categories field not available in the dataset.")
st.divider()

# TOP ENTITÀ ESTRATTE DAL TESTO
st.subheader("🌟 Top Entities")
# Divisione in 3 colonne per mostrare le entità più frequenti
col1, col2, col3 = st.columns([1, 1.07, 1])  # Colonna centrale leggermente più larga

with col1:
    st.subheader("Persons 👥")
    # Top 5 persone più menzionate negli articoli filtrati
    top_persons = get_top_entities_filtered(filters_key, st.session_state.filters, "persons")
    if not top_persons.empty:
        st.dataframe(top_persons, height=212, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for persons.")

with col2:
    st.subheader("Organizations 🏢")
    # Top 5 organizzazioni più menzionate negli articoli filtrati
    top_orgs = get_top_entities_filtered(filters_key, st.session_state.filters, "organizations")
    if not top_orgs.empty:
        st.dataframe(top_orgs, height=212, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for organizations.")

with col3:
    st.subheader("Locations 📍")
    # Top 5 luoghi più menzionati negli articoli filtrati
    top_locations = get_top_entities_filtered(filters_key, st.session_state.filters, "locations")
    if not top_locations.empty:
        st.dataframe(top_locations, height=212, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for locations.")

# Didascalia finale per spiegare l'origine delle entità
st.caption("Top 5 entities (persons, organizations, locations) extracted from the filtered articles' texts.")
