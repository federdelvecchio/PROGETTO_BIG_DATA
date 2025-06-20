from pymongo import MongoClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import string
import nltk

from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Scarica i dati necessari di NLTK (esegui una sola volta)
nltk.download('punkt')
nltk.download('stopwords')

# Connessione a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Funzione per creare un riassunto basato su estrazione
def extractive_summary(text, language="english", num_sentences=3):
    try:
        # Tokenizza il testo in frasi
        sentences = sent_tokenize(text, language=language)
    except LookupError:
        # Fallback: Tokenizzazione semplice per lingue non supportate
        sentences = text.split(". ")  # Divide il testo in base ai punti

    if len(sentences) <= num_sentences:
        return text  # Se il testo ha meno frasi del limite, restituisci tutto il testo

    # Tokenizza le parole e calcola la frequenza
    try:
        words = word_tokenize(text.lower(), language=language) # Converte il testo in minuscolo e suddivide in parole
    except LookupError:
        # Fallback: Tokenizzazione semplice per lingue non supportate
        words = text.lower().split() # Divide il testo in base agli spazi

    # Crea il set delle parole da ignorare (articoli, preposizioni, punteggiatura)
    stop_words = set(stopwords.words(language) + list(string.punctuation))
    # Filtra le parole mantenendo solo quelle significative
    filtered_words = [word for word in words if word not in stop_words]
    # Conta la frequenza di ogni parola 'importante'
    word_frequencies = Counter(filtered_words)

    # Calcola il punteggio di ogni frase
    sentence_scores = {}
    for sentence in sentences: # Itera su ogni frase
        for word in word_tokenize(sentence.lower()): # Tokenizza la frase in parole
            if word in word_frequencies: # Se la parola è significativa
                if sentence not in sentence_scores: # Inizializza il punteggio della frase se non esiste
                    sentence_scores[sentence] = 0
                sentence_scores[sentence] += word_frequencies[word] # Aggiungi la frequenza della parola al punteggio della frase

    # Ordina in modo decrescente le frasi in base al loro punteggio
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    # Seleziona le prime 'num_sentences' frasi per il riassunto e le unisce in un'unica stringa (separatore: spazio)
    summary = " ".join(sorted_sentences[:num_sentences])
    return summary

# Seleziona i documenti che hanno 'text_processed' ma non 'summary'
query = {"text_processed": {"$exists": True}, "summary": {"$exists": False}} 
batch_size = 100  # Numero di documenti da processare per batch

# Calcola il numero totale di documenti
total_docs = collection.count_documents(query)

# Barra di avanzamento
with tqdm(total=total_docs, desc="Creazione delle sintesi", unit="doc") as pbar:
    cursor = collection.find(query, projection={"text_processed": 1, "language": 1}, batch_size=batch_size) # Seleziona solo i campi necessari
    for doc in cursor: # Itera sui documenti
        try:
            text_processed = doc["text_processed"]
            language = doc.get("language", "english")  # Default a "english" se la lingua non è specificata
            
            # Genera il riassunto
            summary = extractive_summary(text_processed, language=language, num_sentences=3) # Numero di frasi nel riassunto
            
            # Aggiorna il documento con il riassunto
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"summary": summary}}
            )
            pbar.update(1)
            
        except Exception as e:
            print(f"Errore durante la creazione del summary per il documento {doc['_id']}: {e}")
            continue

print(f"Campo 'summary' aggiunto a {total_docs} documenti.")
