import streamlit as st
from pymongo import MongoClient
import pandas as pd
import datetime
import sys
import os

# Importa le funzioni utility per il fact checking e le configurazioni
from utils.fact_checker_utils import search_chunks, generate_llm_response, clean_answer, AVAILABLE_MODELS, calculate_result_score
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Titolo principale dell'applicazione
st.title("Fact Checker")

# Sezione per configurare i parametri del modello LLM
with st.expander("⚙️ Model Configuration", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        # Selezione del modello LLM da utilizzare
        selected_model = st.selectbox(
            "Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            key="model_select"
        )
        st.write("")
        # Impostazione del range di token per la risposta
        token_range = st.slider(
            "Token Range (Min - Max)",
            min_value=50,
            max_value=1500,
            value=(100, 500),
            step=25,
            key="token_range"
        )
        min_tokens, max_tokens = token_range
        st.session_state.min_tokens = min_tokens
        st.session_state.max_tokens = max_tokens
    with col2:
        # Selezione del numero di chunk da recuperare
        num_chunks = st.slider(
            "Number of Chunks",
            min_value=5,
            max_value=30, 
            value=5,
            step=1,
            key="num_chunks"
        )
        # Impostazione della temperatura del modello (creatività)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature"
        )

# Sezione per l'inserimento dell'affermazione da verificare
st.markdown("### 📝 Claim to verify:")

claim = st.text_area(
    label="Claim input",
    value="",
    height=75,
    placeholder="Write your claim here...",
    label_visibility="collapsed"
)

st.markdown("")

# Pulsante per avviare la verifica dell'affermazione
col_empty, col_btn = st.columns([5, 1])
with col_btn:
    check_pressed = st.button("Check Claim", type="primary")

# Logica di esecuzione quando viene premuto il pulsante di verifica
if check_pressed:
    if claim:
        with st.spinner("Checking claim...", show_time=True):
            # Recupera i chunk di testo rilevanti dal database
            search_results = search_chunks(claim, st_session=st.session_state)
            if search_results:
                # Connessione al database MongoDB
                client = MongoClient(MONGO_URI)
                db = client[DB_NAME]
                articles_col = db[COLLECTION_NAME]

                # Genera la risposta utilizzando il modello LLM
                llm_response = generate_llm_response(claim, search_results, selected_model)
                
                # Pulizia del testo della risposta
                answer_text = clean_answer(str(llm_response))

                # Recupera i dati completi degli articoli dal database
                article_ids = list(set([result["article_id"] for result in search_results]))
                articles_cursor = articles_col.find({"_id": {"$in": article_ids}})
                articles_data = {article["_id"]: article for article in articles_cursor}
                
                # Prepara i dati per il report testuale
                from datetime import datetime as dt
                current_datetime = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Calcola il punteggio di affidabilità del risultato
                result_score = calculate_result_score(search_results)
                score_percentage = int(result_score * 100)
                
                # Formatta il contenuto del report
                txt_content = f"""FACT CHECK REPORT
================

CLAIM: {claim}

EVALUATION: "{clean_answer(answer_text)}"

RELIABILITY SCORE: {score_percentage}

DATE/TIME: {current_datetime}
                """
                # Salva tutti i dati nella session_state per uso futuro
                st.session_state['last_claim'] = claim
                st.session_state['last_answer'] = answer_text
                st.session_state['last_search_results'] = search_results
                st.session_state['articles_data'] = articles_data
                st.session_state['last_txt_data'] = txt_content.encode("utf-8")
                
                client.close()
            else:
                st.error("No articles found for this claim.")

# Visualizza i risultati se esistono nella session_state (anche dopo refresh)
if 'last_claim' in st.session_state and 'last_answer' in st.session_state:
    claim = st.session_state['last_claim']
    answer_text = st.session_state['last_answer']
    search_results = st.session_state['last_search_results']
    articles_data = st.session_state.get('articles_data', {})
    
    if search_results:
        # Calcola il punteggio di affidabilità
        result_score = calculate_result_score(search_results)
        
        # Visualizza il verdetto in un box stilizzato
        st.markdown(
            f"""
            <div style="background: #222; border: 2px solid #29B6F6; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 28px;">🎯</span>
                    <span style="font-weight: bold; color: #29B6F6; font-size: 22px; margin-left: 12px;">VERDICT</span>
                </div>
                <div style="font-size: 16px; color: white; line-height: 1.6;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        # Visualizza il punteggio di affidabilità con colori diversi in base al valore
        if result_score >= 0.7:
            score_color = "#4CAF50"  # Verde per alta affidabilità
            score_label = "HIGH RELIABILITY"
            score_icon = "✅"
        elif result_score >= 0.4:
            score_color = "#FF9800"  # Arancione per media affidabilità
            score_label = "MEDIUM RELIABILITY"
            score_icon = "⚠️"
        else:
            score_color = "#F44336"  # Rosso per bassa affidabilità
            score_label = "LOW RELIABILITY"
            score_icon = "❌"
        
        score_percentage = int(result_score * 100)
        
        # Visualizza il box con il punteggio di affidabilità
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1a1a1a, #2d2d2d); border: 3px solid {score_color}; border-radius: 12px; padding: 1.2rem; margin: 1rem 0; text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                    <div>
                        <span style="font-size: 32px;">{score_icon}</span>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #BBB; margin-bottom: 6px;">RELIABILITY SCORE</div>
                        <div style="font-size: 28px; font-weight: bold; color: {score_color};">{score_percentage}%</div>
                        <div style="font-size: 12px; color: {score_color}; margin-top: 4px; font-weight: bold;">{score_label}</div>
                        <div style="font-size: 10px; color: #999; margin-top: 2px;">Based on source quality and relevance</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        # Sezione dei risultati della ricerca
        st.markdown("### 📚 Search Results:")
        shown_articles = set()
        articles_shown_count = 0
        
        # Definizione dei colori per i vari articoli
        bright_colors = ["#00E676", "#FF7043", "#29B6F6", "#AB47BC", "#FF4081", "#66BB6A", "#FFB300", "#FFD600"]

        # Visualizza ogni articolo trovato
        for i, result in enumerate(search_results):
            article_id = result["article_id"]
            # Evita di mostrare articoli duplicati
            if article_id in shown_articles:
                continue
            shown_articles.add(article_id)
            
            # Assegna un colore ciclico a ogni articolo
            article_color = bright_colors[articles_shown_count % len(bright_colors)]
            articles_shown_count += 1
            
            # Raccogli tutti i punteggi di rilevanza per i chunk di questo articolo
            article_scores = []
            for chunk_result in search_results:
                if chunk_result["article_id"] == article_id:
                    article_scores.append(round(chunk_result["score"], 2))
            
            # Ordina i punteggi in ordine decrescente
            article_scores.sort(reverse=True)
            scores_str = ", ".join(map(str, article_scores))
            
            # Recupera i dati dell'articolo
            article = articles_data.get(article_id, {})
            
            # Estrai i metadati dell'articolo
            domain_rank = article.get("domain_rank", "N/A")
            url = article.get("url", None)
            summary = article.get("summary", "No summary available")
            title = article.get("title", "Untitled")
            language = article.get("language", "N/A")
            author = article.get("author", "N/A")
            sentiment = article.get("sentiment", "N/A")
            published = article.get("published", "N/A")

            # Formatta la data di pubblicazione in modo leggibile
            if published != "N/A" and published:
                try:
                    from datetime import datetime
                    # Parse della data ISO
                    if isinstance(published, str):
                        # Rimuove il timezone per semplicità
                        clean_date = published.split('T')[0]
                        date_obj = datetime.strptime(clean_date, '%Y-%m-%d')
                        published_formatted = date_obj.strftime('%d %B %Y')
                    else:
                        published_formatted = str(published)
                except:
                    published_formatted = str(published)
            else:
                published_formatted = "N/A"

            # Estrai altre informazioni dell'articolo
            categories = article.get("categories", [])
            country = article.get("country", "N/A")
            
            # Formatta le categorie
            if isinstance(categories, list):
                categories_str = ', '.join(categories) if categories else 'None'
            else:
                categories_str = str(categories) if categories else 'None'
            
            # Estrai le entità dall'articolo
            locations = []
            organizations = []
            persons = []
            if "entities_processed" in article:
                entities = article["entities_processed"]
                locations = sorted(list(set(entities.get("locations", []))))
                organizations = sorted(list(set(entities.get("organizations", []))))
                persons = sorted(list(set(entities.get("persons", []))))

            # Estrai il dominio dall'URL
            import urllib.parse
            if url:
                parsed_url = urllib.parse.urlparse(url)
                domain = parsed_url.netloc
                # Rimuove 'www.' se presente
                if domain.startswith('www.'):
                    domain = domain[4:]
            else:
                domain = "N/A"
            
            # Visualizza l'intestazione dell'articolo
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="width: 120px; flex-shrink: 0;">
                            <span style="background: {article_color}; color: black; padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 16px;">
                                Article {articles_shown_count}
                            </span>
                        </div>
                        <div style="flex-grow: 1;">
                            <h4 style="margin: 0; color: {article_color}; font-size: 20px; font-weight: 700;">{title}</h4>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Visualizza il riepilogo dell'articolo
            st.markdown(
                f"""
                <div style="background: #222; border: 2px solid {article_color}; border-radius: 10px; padding: 1.2rem; margin-bottom: 10px;">
                    <div style="font-size: 15px; color: white; line-height: 1.5;">{summary}</div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Visualizza la fonte e i punteggi
            if url:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: {article_color}; font-weight: bold;">🔗 Source:</span>
                            <a href="{url}" target="_blank" style="color: #29B6F6; text-decoration: none;">{domain}</a>
                        </div>
                        <div style="background: #444; padding: 6px 12px; border-radius: 6px;">
                            <span style="color: #FFD600; font-weight: bold; font-size: 14px;">Scores:</span>
                            <span style="color: white; font-size: 14px; margin-left: 8px;">{scores_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
            
            # Sezione espandibile con informazioni aggiuntive
            with st.expander("🔍 More Information", expanded=False):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    # Visualizza i metadati dell'articolo
                    st.markdown(
                        f"""
                        <div style="background: #333; padding: 1rem; border-radius: 8px; margin-bottom: 10px; height: 250px; overflow-y: auto;">
                            <b style="color: #FFD600;">📊 Metadata</b><br>
                            <b style="color: #29B6F6;">Domain Rank:</b> <span style="color: white;">{domain_rank}</span><br>
                            <b style="color: #29B6F6;">Published:</b> <span style="color: white;">{published_formatted}</span><br>
                            <b style="color: #29B6F6;">Author:</b> <span style="color: white;">{author}</span><br>
                            <b style="color: #29B6F6;">Language:</b> <span style="color: white;">{language}</span><br>
                            <b style="color: #29B6F6;">Country:</b> <span style="color: white;">{country}</span><br>
                            <b style="color: #29B6F6;">Categories:</b> <span style="color: white;">{categories_str}</span><br>
                            <b style="color: #29B6F6;">Sentiment:</b> <span style="color: white;">{sentiment}</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with col_info2:
                    # Visualizza le entità estratte dall'articolo
                    st.markdown(
                        f"""
                        <div style="background: #333; padding: 1rem; border-radius: 8px; margin-bottom: 10px; height: 250px; overflow-y: scroll;">
                            <b style="color: #FFD600;">🏷️ Entities</b><br>
                            <b style="color: #FF7043;">Organizations:</b> <span style="color: white;">{', '.join(organizations) if organizations else 'None'}</span><br>
                            <b style="color: #66BB6A;">Locations:</b> <span style="color: white;">{', '.join(locations) if locations else 'None'}</span><br>
                            <b style="color: #AB47BC;">Persons:</b> <span style="color: white;">{', '.join(persons) if persons else 'None'}</span>
                        </div>
                        """, unsafe_allow_html=True
                    )

        # Sezione per i pulsanti di download
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        col1, col_center, col3 = st.columns([0.5, 2, 0.5])
        with col_center:
            col_btn1, col_btn2 = st.columns(2)
            
            # Pulsante per scaricare il report in formato TXT
            with col_btn1:
                if 'last_txt_data' in st.session_state:
                    st.download_button(
                        label="📈 Download Report",
                        data=st.session_state['last_txt_data'],
                        file_name="fact_check_report.txt",
                        mime="text/plain",
                        type="secondary",
                        use_container_width=True
                    )
            
            # Pulsante per scaricare i dati degli articoli in formato CSV
            with col_btn2:
                if articles_data:
                    # Prepara i dati per il CSV
                    csv_rows = []
                    for article_id, article in articles_data.items():
                        if article:
                            # Estrae tutte le informazioni dell'articolo
                            row = {
                                'id': str(article.get('_id', '')),
                                'uuid': article.get('uuid', ''),
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'author': article.get('author', ''),
                                'published': article.get('published', ''),
                                'text': article.get('text', ''),
                                'text_processed': article.get('text_processed', ''),
                                'language': article.get('language', ''),
                                'sentiment': article.get('sentiment', ''),
                                'country': article.get('country', ''),
                                'domain_rank': article.get('domain_rank', ''),
                                'domain_rank_updated': article.get('domain_rank_updated', ''),
                                'site': article.get('site', ''),
                                'site_type': article.get('site_type', ''),
                                'categories': ', '.join(article.get('categories', [])) if isinstance(article.get('categories'), list) else str(article.get('categories', '')),
                                'summary': article.get('summary', ''),
                                'main_image': article.get('main_image', ''),
                                'crawled': article.get('crawled', ''),
                                'updated': article.get('updated', ''),
                                'site_categories': ', '.join(article.get('site_categories', [])) if isinstance(article.get('site_categories'), list) else str(article.get('site_categories', '')),
                                'vectorized': article.get('vectorized', '')
                            }
                            
                            # Aggiunge le entità estratte (gestisce sia il formato originale che quello processato)
                            if 'entities_processed' in article:
                                entities = article['entities_processed']
                                row['persons'] = ', '.join(entities.get('persons', []))
                                row['organizations'] = ', '.join(entities.get('organizations', []))
                                row['locations'] = ', '.join(entities.get('locations', []))
                                row['persons_sentiment'] = ''
                                row['organizations_sentiment'] = ''
                                row['locations_sentiment'] = ''
                            elif 'entities' in article:
                                entities = article['entities']
                                row['persons'] = ', '.join([p['name'] for p in entities.get('persons', [])])
                                row['organizations'] = ', '.join([o['name'] for o in entities.get('organizations', [])])
                                row['locations'] = ', '.join([l['name'] for l in entities.get('locations', [])])
                                row['persons_sentiment'] = ', '.join([f"{p['name']}({p['sentiment']})" for p in entities.get('persons', [])])
                                row['organizations_sentiment'] = ', '.join([f"{o['name']}({o['sentiment']})" for o in entities.get('organizations', [])])
                                row['locations_sentiment'] = ', '.join([f"{l['name']}({l['sentiment']})" for l in entities.get('locations', [])])
                            else:
                                row['persons'] = ''
                                row['organizations'] = ''
                                row['locations'] = ''
                                row['persons_sentiment'] = ''
                                row['organizations_sentiment'] = ''
                                row['locations_sentiment'] = ''
                            

                            csv_rows.append(row)
                    
                    # Converte in DataFrame e poi in CSV
                    df_articles = pd.DataFrame(csv_rows)
                    csv_data = df_articles.to_csv(index=False, encoding='utf-8')
                    
                    # Pulsante per il download del CSV
                    st.download_button(
                        label="📰 Download Articles CSV",
                        data=csv_data.encode('utf-8'),
                        file_name="fact_check_articles.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True
                    )

    else:
        # Messaggio quando non vengono trovati articoli
        st.markdown(
            """
            <div style="background: #4A90E2; border-radius: 8px; padding: 1rem; text-align: center;">
                <span style="font-size: 20px;"></span>
                <span style="color: white; font-weight: bold; margin-left: 8px;">No articles found for this claim.</span>
            </div>
            """, unsafe_allow_html=True
        )