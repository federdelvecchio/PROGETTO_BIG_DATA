import os
import json
import time
from pymongo import MongoClient, ASCENDING, errors
from tqdm import tqdm

from config import MONGO_URI, DB_NAME, COLLECTION_NAME, FIELDS_MAP

DATASET_DIR = "News_Datasets"

def get_all_json_files():
    """
    Questa funzione restituisce tutti i path dei file .json presenti 
    nella cartella e nelle sottocartelle di DATASET_DIR.
    
    Yields:
        str: Percorso completo di ogni file JSON trovato
    """
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".json"):
                yield os.path.join(root, file)

def extract_field(doc, *paths):
    """
    Estrae un campo da un JSON annidato seguendo percorsi alternativi.
    
    Lo script gestisce la duplicazione tra il livello principale e l'oggetto thread,
    provando multipli percorsi per trovare il valore desiderato. Questo assicura
    un'estrazione robusta anche con  JSON variabili.
    
    Args:
        doc (dict): Il documento JSON da cui estrarre il campo
        *paths (str): Percorsi alternativi separati da punti (es. "thread.title", "title")
    
    Returns:
        Il valore trovato nel primo percorso valido, None se nessun percorso è valido
    """
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

def build_flat_doc(doc):
    """
    Appiattisce un documento JSON complesso in una struttura semplificata.
    
    Utilizza la mappatura FIELDS_MAP per selezionare diciotto campi chiave
    dal JSON originale, trasformando la struttura annidata in un documento
    ottimizzato per l'inserimento in MongoDB.
    
    Args:
        doc (dict): Il documento JSON originale da appiattire
        
    Returns:
        dict: Documento con struttura semplificata contenente solo i campi mappati
    """
    flat = {}
    for out_key, in_paths in FIELDS_MAP.items():
        flat[out_key] = extract_field(doc, *in_paths)
    return flat

if __name__ == "__main__":
    # 1. Prova a connettersi a MongoDB e verifica la connessione.
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # 
        client.admin.command('ping') # Verifica la connessione
        print("Connessione a MongoDB riuscita.")
    except errors.ServerSelectionTimeoutError as err:
        print(f"Connessione a MongoDB fallita: {err}")
        exit(1)  # Termina lo script in caso di errore

    # 2. Imposta il database e la collezione
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 3. Crea un indice unico sul campo 'url' se non esiste già.
    if "url_1" not in collection.index_information():
        collection.create_index([("url", ASCENDING)], unique=True) 
        print("Indice unico su 'url' creato.")

    # 4. Ottieni la lista di tutti i file JSON da importare.
    json_files = list(get_all_json_files())
    inserted, skipped = 0, 0 # Contatori per documenti inseriti e saltati
    print("Inserimento dei documenti in MongoDB in corso...")

    start_time = time.time() # Inizia il timer per calcolare il tempo impiegato
    # 5. Cicla su tutti i file JSON e importa i dati.
    for json_path in tqdm(json_files, desc="Avanzamento", unit="file"): # Barra di avanzamento
        with open(json_path, encoding="utf-8") as f: 
            doc = json.load(f) # Carica il documento JSON
        flat_doc = build_flat_doc(doc) # Appiattisce il documento
        url = flat_doc.get("url")
        if not url: # Se manca la URL, salta il documento.
            continue
        try:
            collection.insert_one(flat_doc) # Inserisce il documento
            inserted += 1 # Incrementa il contatore
        except Exception as e:
            # Se il documento è duplicato (stessa URL), incrementa skipped.
            if "duplicate key error" in str(e):
                skipped += 1
            else:
                print(f"Errore su {json_path}: {e}")
    elapsed = time.time() - start_time # Calcola il tempo impiegato
    print(f"Inseriti: {inserted}, Saltati (duplicati): {skipped}")
    print(f"Tempo impiegato: {elapsed:.2f} secondi")