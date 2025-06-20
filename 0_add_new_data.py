import requests
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from bs4 import BeautifulSoup

from config import MONGO_URI, DB_NAME, COLLECTION_NAME, FIELDS_MAP

API_KEY = st.secrets["webz_key"]
MAX_ARTICLES = 10
LOG_FILE = "added_articles_log.txt"

# Funzione per estrarre un campo da un dizionario annidato, seguendo uno o più "percorsi" (path)
def extract_field(doc, *paths):
    for path in paths:
        keys = path.split('.')
        value = doc
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        if value is not None:
            return value
    return None

# Funzione che "appiattisce" un documento secondo la mappatura FIELDS_MAP
def build_flat_doc(doc):
    flat = {}
    for out_key, in_paths in FIELDS_MAP.items():
        flat[out_key] = extract_field(doc, *in_paths)
    return flat

# Funzione che interroga l'API per scaricare articoli (fino a MAX_ARTICLES)
def fetch_articles(size=MAX_ARTICLES):
    url = "https://api.webz.io/newsApiLite"
    params = { # Parametri per la richiesta all'API
        "token": API_KEY,
        "size": size,
        "q": "news"
    }
    response = requests.get(url, params=params) 
    if response.status_code != 200: # Controlla se la richiesta ha avuto successo
        print("Errore API:", response.text)
    response.raise_for_status() # Controlla se la risposta è stata correttamente ricevuta
    data = response.json() # Controlla se la risposta è in formato JSON
    return data.get("articles", data.get("posts", [])) # Restituisce gli articoli o i post

# Funzione per eseguire lo scraping del testo da un URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10) 
        response.raise_for_status()
        # Usa BeautifulSoup per analizzare il contenuto HTML
        soup = BeautifulSoup(response.text, 'html.parser') 
        # Prova a estrarre il testo principale
        paragraphs = soup.find_all('p') # Trova tutti i paragrafi nel documento HTML
        text = ' '.join([p.get_text() for p in paragraphs]) # Unisce il testo di tutti i paragrafi in un'unica stringa
        return text.strip() # Rimuove gli spazi iniziali e finali dal testo
    except Exception as e:
        print(f"Errore durante lo scraping dell'URL {url}: {e}")
        return "Errore durante lo scraping del testo"

# Funzione che controlla se un articolo con una certa URL è già presente nel database
def article_exists(collection, url):
    return collection.count_documents({"url": url}, limit=1) > 0

def main():
    try:
        # Connessione a MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        collection = client[DB_NAME][COLLECTION_NAME]

        # Scarica gli articoli dall'API
        articles = fetch_articles()
        total = len(articles) # Numero totale di articoli scaricati
        print(f"Articoli scaricati: {total}")

        # Lista per raccogliere gli _id degli articoli aggiunti
        added_ids = []

        # Processa gli articoli
        for article in articles:
            # Estrai la URL
            url = extract_field(article, "url", "thread.url")
            if not url: # Se non c'è una URL, salta l'articolo
                print("URL non trovato, salto l'articolo.")
                continue

            # Esegui lo scraping del testo
            scraped_text = scrape_text_from_url(url)

            # Sostituisci il campo "text" con il testo recuperato tramite scraping
            if article.get("text") == "Full text is unavailable in the news API lite version":
                article["text"] = scraped_text

            # Appiattisci il documento secondo la mappatura
            flat_doc = build_flat_doc(article)

            # Inserisci il documento nel database
            result = collection.insert_one(flat_doc)
            added_ids.append(result.inserted_id)

        # Scrivi gli _id degli articoli aggiunti nel log
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"Articoli aggiunti il {datetime.now()}:\n")
            for _id in added_ids:
                log_file.write(f" - {_id}\n")
            log_file.write("==================================================\n")

        # Stampa gli _id degli articoli aggiunti
        print("ID degli articoli aggiunti:")
        for _id in added_ids:
            print(f" - {_id}")

    except Exception as e:
        print("Errore durante l'esecuzione:", e)

if __name__ == "__main__":
    main()