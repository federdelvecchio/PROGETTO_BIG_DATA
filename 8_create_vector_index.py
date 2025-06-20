from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from config import MONGO_URI, DB_NAME

# Connessione al database MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
chunks_col = db["article_chunks"]  # Collezione contenente i chunk di testo e relativi embedding

# Definizione del modello dell'indice vettoriale per la ricerca semantica
# Questo indice consente di effettuare query di similarità vettoriale sui chunk
index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",  # Campo embedding generato in 7_chunk_and_embed.py
                "numDimensions": 384,  # Dimensione del vettore (384 per il modello paraphrase-multilingual-MiniLM-L12-v2)
                "similarity": "cosine"  # Metodo di calcolo della similarità (coseno dell'angolo tra vettori)
                                        # Alternative: "dotProduct" (prodotto scalare) o "euclidean" (distanza euclidea)
            }
        ]
    },
    name="vector_index",  # Nome dell'indice per riferimenti futuri nelle query
    type="vectorSearch"   # Tipo specifico di indice per ricerche vettoriali in MongoDB Atlas
)

# Creazione dell'indice nella collezione
# Questo processo può richiedere tempo in base alla dimensione della collezione
# Nota: richiede MongoDB Atlas con funzionalità di ricerca vettoriale abilitata
chunks_col.create_search_index(model=index_model)

# Messaggio di conferma della creazione dell'indice
print("Indice vettoriale creato con successo nella collezione 'article_chunks'.")