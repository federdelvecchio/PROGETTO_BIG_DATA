import os
from pymongo import MongoClient
from tqdm import tqdm
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parametri di configurazione per il chunking
CHUNK_SIZE = 500          # Dimensione in caratteri di ogni chunk
CHUNK_OVERLAP = 100       # Sovrapposizione tra chunk consecutivi per mantenere il contesto

# Selezione e inizializzazione del modello di embedding
# Il modello multilingue supporta sia l'italiano che le altre lingue
#EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Versione alternativa (solo inglese)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Inizializzazione dello splitter per dividere i testi in chunk
# Utilizza un approccio ricorsivo che prova diversi separatori in ordine
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]  # Prova a dividere prima per paragrafi, poi per righe, parole e caratteri
)

if __name__ == "__main__":
    # Connessione al database MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    articles_col = db[COLLECTION_NAME]
    chunks_col = db["article_chunks"]  # Collezione per i chunk degli articoli

    # Query per selezionare solo i documenti che devono essere processati
    # Seleziona articoli con testo elaborato ma non ancora vettorizzati
    query = {"text_processed": {"$exists": True}, "vectorized": {"$exists": False}}
    docs = list(articles_col.find(query, {"_id": 1, "text_processed": 1}))

    # Elaborazione di ogni documento
    for doc in tqdm(docs, desc="Chunking and embedding", unit="doc"):
        # Estrazione del testo processato
        text = doc.get("text_processed", "")
        
        # Divisione del testo in chunk
        chunks = splitter.split_text(text)
        if not chunks:
            continue  # Salta documenti senza chunk validi

        # Generazione degli embedding per tutti i chunk dell'articolo
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Preparazione dei documenti per i chunk da inserire nel database
        chunk_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_docs.append({
                "article_id": doc["_id"],      # Riferimento all'articolo originale
                "chunk_index": i,              # Posizione del chunk nell'articolo
                "chunk_text": chunk,           # Testo del chunk
                "embedding": embedding         # Vettore di embedding per la ricerca semantica
            })
        
        # Inserimento dei chunk nella collezione dedicata
        chunks_col.insert_many(chunk_docs)

        # Aggiornamento dell'articolo originale per indicare che è stato vettorizzato
        # Evita di rielaborare lo stesso articolo in esecuzioni future
        articles_col.update_one({"_id": doc["_id"]}, {"$set": {"vectorized": True}})

    # Messaggio di completamento
    print(f"Chunking ed embedding completati per {len(docs)} documenti.")

