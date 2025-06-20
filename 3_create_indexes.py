from pymongo import MongoClient, ASCENDING, errors
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# 1. Connessione a MongoDB e selezione del database e della collection.
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# 2. Lista dei campi su cui creare l'indice (se non esiste già).
fields_to_index = [
    "uuid",
    "country",
    "language",
    "published",
    "sentiment",
    "categories",
    "domain_rank",
    "site",
    "site_type",
    "site_categories",
    "categories",
    "updated"
]

# 3. Ciclo su tutti i campi e creazione degli indici.
for field in fields_to_index:
    try:
        # Crea un indice unico per il campo 'uuid', altrimenti un indice normale.
        if field == "uuid":
            collection.create_index([(field, ASCENDING)], unique=True)
        else:
            collection.create_index([(field, ASCENDING)])
    except errors.OperationFailure as e:
        # Se l'indice esiste già con opzioni diverse, ignora e continua.
        print(f"Indice già esistente o errore su '{field}': {e}")

print("Indici assicurati per i campi:", fields_to_index)