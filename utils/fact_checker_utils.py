import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import streamlit as st
import ast
import google.generativeai as genai
from huggingface_hub import login

# Costanti per la configurazione del sistema
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Elenco dei modelli LLM disponibili per il fact checking
AVAILABLE_MODELS = {
    "gemma-3-1b-it (local)": "google/gemma-3-1b-it",
    "gemini-2.0-flash (api)": "gemini-2.0-flash",
}

@st.cache_resource(show_spinner=True)
def initialize_models():
    """
    Inizializza i modelli di embedding e LLM utilizzati dal sistema.
    Usa la cache di Streamlit per evitare di ricaricare i modelli ad ogni refresh.
    
    Returns:
        tuple: (modelli caricati, dispositivo utilizzato)
    """
    st.write("🔄 DEBUG: Inizializzazione modelli...")
    login(st.secrets["hugging_face_token"])
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"🖥️ DEBUG: Dispositivo utilizzato: {device}")
    
    # Inizializzazione del modello di embedding per la ricerca semantica
    st.write("🔍 DEBUG: Caricamento modello di embedding...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL).to(device)
    models['embedding_model'] = embedding_model
    st.write("✅ DEBUG: Modello di embedding caricato")

    # Inizializzazione dei modelli LLM locali (senza quantizzazione per compatibilità CPU)
    models['llm'] = {}
    for model_name, model_path in AVAILABLE_MODELS.items():
        if "gemini" in model_name.lower():
            st.write(f"⏭️ DEBUG: Saltando modello API: {model_name}")
            continue  # Salta i modelli API che non richiedono inizializzazione locale
        
        st.write(f"🤖 DEBUG: Caricamento modello LLM: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Aggiunge il token di padding se mancante
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            st.write(f"🔧 DEBUG: Aggiunto pad_token per {model_name}")
        
        model_llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Configurazione compatibile con CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        pipeline_llm = pipeline(
            "text-generation",
            model=model_llm,
            tokenizer=tokenizer,
            device=device
        )
        models['llm'][model_name] = {
            'tokenizer': tokenizer,
            'model': model_llm,
            'pipeline': pipeline_llm
        }
        st.write(f"✅ DEBUG: Modello {model_name} caricato con successo")
    
    st.write("🎉 DEBUG: Inizializzazione modelli completata")
    return models, device

# Inizializzazione dei modelli all'avvio dell'applicazione
models, device = initialize_models()

@st.cache_data(ttl=3600, show_spinner=True)
def get_domain_rank_threshold():
    """
    Calcola la soglia del 75° percentile per il domain_rank degli articoli.
    Utilizzato per filtrare le fonti meno affidabili.
    
    Returns:
        float: Valore di soglia per il domain_rank (75° percentile)
    """
    st.write("📊 DEBUG: Calcolo soglia domain_rank...")
    from config import MONGO_URI, DB_NAME, COLLECTION_NAME
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    articles_col = db[COLLECTION_NAME]
    
    # Pipeline di aggregazione per calcolare il 75° percentile
    pipeline = [
        {"$match": {"domain_rank": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": None, "domain_ranks": {"$push": "$domain_rank"}}},
        {"$project": {
            "percentile_75": {
                "$arrayElemAt": [
                    {"$sortArray": {"input": "$domain_ranks", "sortBy": 1}},
                    {"$floor": {"$multiply": [{"$size": "$domain_ranks"}, 0.75]}}
                ]
            }
        }}
    ]
    
    result = list(articles_col.aggregate(pipeline))
    threshold = result[0]["percentile_75"] if result else 1000000
    st.write(f"📈 DEBUG: Soglia domain_rank calcolata: {threshold}")
    
    client.close()
    return threshold

def search_chunks(query, top_k=None, st_session=None):
    """
    Esegue la ricerca semantica dei chunk di testo nel database dato un claim.
    Applica filtri di qualità basati su score di similarità e affidabilità della fonte.
    
    Args:
        query (str): Il claim da verificare
        top_k (int, optional): Numero di chunk da recuperare
        st_session (object, optional): Sessione Streamlit per accedere ai parametri
    
    Returns:
        list: Lista dei chunk di testo più rilevanti con relativi metadati
    """
    st.write(f"🔍 DEBUG: Ricerca chunk per query: '{query[:50]}...'")
    
    if st_session and top_k is None:
        top_k = st_session.get('num_chunks', 5)
    elif top_k is None:
        top_k = 5

    st.write(f"📊 DEBUG: Numero di chunk richiesti: {top_k}")

    embedding_model = models['embedding_model']
    st.write("🔄 DEBUG: Generazione embedding per la query...")
    query_embedding = embedding_model.encode(query).tolist()
    st.write(f"✅ DEBUG: Embedding generato, dimensioni: {len(query_embedding)}")
    
    # Connessione al database MongoDB
    from config import MONGO_URI, DB_NAME, COLLECTION_NAME
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    chunks_col = db["article_chunks"]
    articles_col = db[COLLECTION_NAME]
    st.write("🔗 DEBUG: Connessione al database stabilita")
    
    # Calcola la soglia per il domain_rank
    domain_threshold = get_domain_rank_threshold()
    
    # Ricerca vettoriale iniziale per ottenere i candidati
    st.write("🎯 DEBUG: Esecuzione ricerca vettoriale...")
    initial_results = chunks_col.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,  # Numero elevato di candidati iniziali
                "limit": top_k * 3,    # Risultati extra per il filtraggio successivo
            }
        },
        {
            "$project": {
                "chunk_text": 1,
                "article_id": 1,
                "_id": 0,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ])
    
    initial_results = list(initial_results)
    st.write(f"📈 DEBUG: Risultati iniziali trovati: {len(initial_results)}")
    
    # Filtra i risultati con score di similarità inferiore a 0.75
    score_filtered = [r for r in initial_results if r["score"] >= 0.75]
    st.write(f"🔧 DEBUG: Dopo filtro score >= 0.75: {len(score_filtered)}")
    
    # Recupera il domain_rank per tutti gli articoli filtrati
    article_ids = [r["article_id"] for r in score_filtered]
    articles_info = {}
    if article_ids:
        st.write(f"📋 DEBUG: Recupero informazioni per {len(article_ids)} articoli...")
        articles_cursor = articles_col.find(
            {"_id": {"$in": article_ids}},
            {"_id": 1, "domain_rank": 1}
        )
        for article in articles_cursor:
            articles_info[article["_id"]] = article.get("domain_rank", float('inf'))
        st.write(f"✅ DEBUG: Informazioni recuperate per {len(articles_info)} articoli")
    
    # Filtra per domain_rank (rank più basso = fonte più affidabile)
    final_results = []
    for result in score_filtered:
        article_domain_rank = articles_info.get(result["article_id"], float('inf'))
        if article_domain_rank <= domain_threshold:
            result["domain_rank"] = article_domain_rank
            final_results.append(result)
    
    st.write(f"🎯 DEBUG: Dopo filtro domain_rank <= {domain_threshold}: {len(final_results)}")
    
    # Limita ai top_k risultati finali
    final_results = final_results[:top_k]
    st.write(f"📊 DEBUG: Risultati finali dopo limite top_k: {len(final_results)}")
    
    # Informazioni di debug per la sessione Streamlit
    debug_info = {
        "initial_count": len(initial_results),
        "after_score_filter": len(score_filtered),
        "final_count": len(final_results),
        "domain_threshold": domain_threshold,
        "score_threshold": 0.75
    }
    
    if st_session:
        st_session["debug_info"] = debug_info
        st.write(f"🔍 DEBUG: Info debug salvate in sessione: {debug_info}")
    
    client.close()
    return final_results

@st.cache_data(ttl=3600, show_spinner=False)
def generate_llm_response(claim, _chunks, selected_model):
    """
    Genera una risposta utilizzando un modello LLM per verificare il claim.
    
    Args:
        claim (str): L'affermazione da verificare
        _chunks (list): I chunk di testo recuperati dal database
        selected_model (str): Il nome del modello LLM da utilizzare
    
    Returns:
        str: La risposta generata dal modello LLM
    """
    st.write(f"🤖 DEBUG: Generazione risposta LLM con modello: {selected_model}")
    st.write(f"📊 DEBUG: Numero di chunk ricevuti: {len(_chunks)}")
    
    default_model = list(AVAILABLE_MODELS.keys())[0]
    selected_model = st.session_state.get('model_select', default_model)
    min_tokens = st.session_state.get('min_tokens', 100)
    max_tokens = st.session_state.get('max_tokens', 500)
    temperature = st.session_state.get('temperature', 0.7)

    st.write(f"⚙️ DEBUG: Parametri - min_tokens: {min_tokens}, max_tokens: {max_tokens}, temperature: {temperature}")

    # Costruzione del prompt con numerazione dei chunk
    numbered_chunks = []
    for i, chunk in enumerate(_chunks, 1):
        numbered_chunks.append(f"{i}) {chunk['chunk_text']}")
    context = "\n".join(numbered_chunks)
    prompt = (
        f"Your task: Determine if this claim is TRUE or FALSE (using the context) and provide a detailed explanation.\n\n"
        f"Don't directly reference the context in your answer, but use it to support your reasoning.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    
    st.write(f"📝 DEBUG: Prompt costruito, lunghezza: {len(prompt)} caratteri")

    # Gestione diversa per modelli API vs modelli locali
    if selected_model == "gemini-2.0-flash (api)":
        st.write("🌟 DEBUG: Utilizzando Gemini API...")
        return generate_gemini_response(prompt, min_tokens, max_tokens, temperature)
    else:
        st.write(f"🖥️ DEBUG: Utilizzando modello locale: {selected_model}")
        llm_pipeline = models['llm'][selected_model]['pipeline']
        
        try:
            st.write("🔄 DEBUG: Avvio generazione con modello locale...")
            # Generazione della risposta con il modello locale
            output = llm_pipeline(
                prompt,
                min_new_tokens=min_tokens,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=llm_pipeline.tokenizer.eos_token_id,
                return_full_text=False  # Solo testo generato, non l'intero prompt
            )
            
            # Estrazione del testo generato
            generated_text = output[0]["generated_text"]
            st.write(f"✅ DEBUG: Risposta generata, lunghezza: {len(generated_text)} caratteri")
            
            return generated_text.strip()
            
        except Exception as e:
            st.write(f"❌ DEBUG: Errore durante generazione: {str(e)}")
            st.error(f"Errore nella generazione: {str(e)}")
            return "Errore nella generazione della risposta"

def clean_answer(answer_text):
    """
    Pulisce e formatta la risposta generata dal modello LLM.
    Rimuove caratteri speciali e uniforma la formattazione.
    
    Args:
        answer_text (str): Il testo della risposta da pulire
    
    Returns:
        str: Il testo pulito e formattato
    """
    st.write(f"🧹 DEBUG: Pulizia risposta, lunghezza input: {len(str(answer_text))} caratteri")
    
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    
    # Rimozione di caratteri di markup
    answer_text = answer_text.replace("*", "")
    
    # Gestione corretta dei caratteri di a capo
    answer_text = answer_text.replace("\\n", "\n")
    
    # Rimozione di altri caratteri di escape
    answer_text = answer_text.replace("\\t", "\t")
    
    # Rimozione di token speciali dei modelli
    answer_text = answer_text.replace("<|end_of_text|>", "").replace("<|endoftext|>", "")
    
    # Pulizia degli spazi eccessivi mantenendo la struttura a paragrafi
    lines = answer_text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Rimozione degli spazi all'inizio e alla fine di ogni riga
        cleaned_line = line.strip()
        cleaned_lines.append(cleaned_line)
    
    # Eliminazione delle righe vuote consecutive (max una riga vuota)
    final_lines = []
    prev_empty = False
    
    for line in cleaned_lines:
        if line == "":  # Riga vuota
            if not prev_empty:  # Aggiungi solo se la precedente non era vuota
                final_lines.append(line)
            prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False
    
    clean_text = "\n".join(final_lines)
    st.write(f"✅ DEBUG: Pulizia completata, lunghezza output: {len(clean_text)} caratteri")
    
    return clean_text

def calculate_result_score(search_results):
    """
    Calcola un punteggio di affidabilità basato sui risultati della ricerca.
    Considera sia la rilevanza semantica che l'affidabilità delle fonti.
    
    Formula: sommatoria(score * normalized_rank) / sommatoria(score)
    dove normalized_rank = (max_rank - current_rank) / (max_rank - min_rank)
    
    Args:
        search_results (list): I risultati della ricerca con score e domain_rank
    
    Returns:
        float: Punteggio di affidabilità tra 0.0 e 1.0
    """
    st.write(f"📊 DEBUG: Calcolo punteggio affidabilità per {len(search_results)} risultati")
    
    if not search_results:
        st.write("⚠️ DEBUG: Nessun risultato di ricerca, punteggio = 0.0")
        return 0.0
    
    # Raccolta dei domain_rank validi
    valid_ranks = []
    for result in search_results:
        domain_rank = result.get("domain_rank", float('inf'))
        if domain_rank > 0 and domain_rank != float('inf'):
            valid_ranks.append(domain_rank)
    
    st.write(f"🔢 DEBUG: Domain rank validi trovati: {len(valid_ranks)}")
    
    if not valid_ranks:
        st.write("⚠️ DEBUG: Nessun domain rank valido, punteggio = 0.0")
        return 0.0
    
    # Calcolo di min e max per la normalizzazione
    min_rank = min(valid_ranks)
    max_rank = max(valid_ranks)
    rank_range = max_rank - min_rank
    
    st.write(f"📈 DEBUG: Range domain rank - min: {min_rank}, max: {max_rank}, range: {rank_range}")
    
    # Gestione del caso in cui tutti i rank sono uguali
    if rank_range == 0:
        st.write("⚖️ DEBUG: Tutti i rank sono uguali, punteggio = 1.0")
        return 1.0
    
    numerator = 0.0
    denominator = 0.0
    valid_chunks = 0
    invalid_chunks = 0
    
    for result in search_results:
        score = result.get("score", 0)
        domain_rank = result.get("domain_rank", float('inf'))
        
        if domain_rank > 0 and domain_rank != float('inf'):
            # Normalizzazione: rank migliori (più bassi) → valore più alto
            normalized_rank = (max_rank - domain_rank) / rank_range
            numerator += score * normalized_rank
            denominator += score
            valid_chunks += 1
        else:
            invalid_chunks += 1
    
    final_score = numerator / denominator if denominator > 0 else 0.0
    st.write(f"🎯 DEBUG: Punteggio finale calcolato: {final_score:.3f} (chunk validi: {valid_chunks}, invalidi: {invalid_chunks})")
    
    return final_score

def generate_gemini_response(prompt, min_tokens=100, max_tokens=500, temperature=0.7):
    """
    Genera una risposta utilizzando l'API di Google Gemini.
    
    Args:
        prompt (str): Il prompt da inviare al modello
        min_tokens (int): Numero minimo di token da generare
        max_tokens (int): Numero massimo di token da generare
        temperature (float): Temperatura di generazione (creatività)
    
    Returns:
        str: La risposta generata da Gemini
    """
    st.write("🌟 DEBUG: Inizializzazione API Gemini...")
    
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.write("❌ DEBUG: Gemini API key non trovata")
        st.error("Gemini API key non trovata. Inseriscila in .streamlit/secrets.toml come GEMINI_API_KEY.")
        return "Errore: Gemini API key mancante"
    
    st.write("🔑 DEBUG: API key trovata, configurazione in corso...")
    
    # Configurazione dell'API Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    st.write(f"🔄 DEBUG: Invio richiesta a Gemini (max_tokens: {max_tokens}, temperature: {temperature})...")
    
    try:
        # Generazione della risposta
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        st.write(f"✅ DEBUG: Risposta ricevuta da Gemini, lunghezza: {len(response.text)} caratteri")
        return response.text
    except Exception as e:
        st.write(f"❌ DEBUG: Errore API Gemini: {str(e)}")
        st.error(f"Errore API Gemini: {str(e)}")
        return f"Errore API Gemini: {str(e)}"

