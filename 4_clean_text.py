import time
import re
from pymongo import MongoClient, errors
from tqdm import tqdm
from cleantext import clean
import ftfy

from config import MONGO_URI, DB_NAME, COLLECTION_NAME

LOG_FILE = "pulizia_log.txt"

def clean_article_text(text):
    if not isinstance(text, str):
        return ""
    try:
        # Correggi problemi di encoding e caratteri strani
        text = ftfy.fix_text(text)
        # Pulizia base con clean-text
        text = clean(
            text,
            fix_unicode=True, # Corregge i caratteri Unicode
            lower=False,   # Mantiene il testo in maiuscolo/minuscolo 
            normalize_whitespace=True, # Normalizza gli spazi bianchi
            no_emoji=True, # Rimuove le emoji
            no_line_breaks=True, # Rimuove i ritorni a capo
            to_ascii=True, # Converte in ASCII
            no_urls=True, # Rimuove gli URL
            no_emails=True, # Rimuove le email
            no_phone_numbers=True, # Rimuove i numeri di telefono
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol=""
        )
        # Sostituisci \n, \r, \t con uno spazio
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Riduci sequenze di spazi a uno solo
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        print(f"Errore nella pulizia del testo: {e}")
        return None

def main():
    # 1. Connessione a MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
    except errors.ServerSelectionTimeoutError as err:
        exit(1)

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 2. Seleziona tutti i documenti che hanno il campo 'text' ma non 'text_processed'
    query = {"text": {"$exists": True}, "text_processed": {"$exists": False}}
    docs = collection.find(query, {"_id": 1, "text": 1})
    total = collection.count_documents(query)
    
    # 3. Pulizia e aggiornamento
    start_time = time.time()
    updated = 0
    eliminated = 0
    errors_count = 0

    log_lines = [
        f"LOG PULIZIA ({time.strftime('%Y-%m-%d %H:%M:%S')})\n",
        f"Documenti da processare: {total}\n"
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as logf:  
        for doc in tqdm(docs, total=total, desc="Pulizia testo", unit="doc", dynamic_ncols=True):
            doc_id = doc["_id"]
            try:
                cleaned = clean_article_text(doc.get("text", "")) # Pulizia del testo
                if cleaned is None:
                    errors_count += 1
                    continue
                if len(cleaned) < 50: # Elimina documenti con testo troppo corto
                    collection.delete_one({"_id": doc_id})
                    eliminated += 1
                else: # Aggiorna il documento con il testo pulito
                    result = collection.update_one(
                        {"_id": doc_id},
                        {"$set": {"text_processed": cleaned}} 
                    )
                    if result.modified_count: # Aggiornamento riuscito
                        updated += 1 
                    else:
                        errors_count += 1
            except Exception as e:
                errors_count += 1

        elapsed = time.time() - start_time
        log_lines.append(f"Documenti eliminati (testo < 50 caratteri): {eliminated}\n")
        print(f"Documenti eliminati (testo < 50 caratteri): {eliminated}")
        log_lines.append(f"Documenti aggiornati: {updated}\n")
        print(f"Documenti ripuliti: {updated}")
        log_lines.append(f"Errori: {errors_count}\n")
        log_lines.append(f"Tempo impiegato: {elapsed:.2f} secondi\n")
        logf.writelines(log_lines)

if __name__ == "__main__":
    main()