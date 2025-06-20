from pymongo import MongoClient
import spacy
import spacy.cli
from tqdm import tqdm

from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Mappatura delle lingue ai modelli SpaCy
LANGUAGE_MODELS = {
    "english": "en_core_web_sm",
    "italian": "it_core_news_sm",
    "german": "de_core_news_sm",
    "spanish": "es_core_news_sm",
    "catalan": "ca_core_news_sm",
    "french": "fr_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "dutch": "nl_core_news_sm",
    "chinese": "zh_core_web_sm"
}
# Cache per i modelli SpaCy già caricati
loaded_models = {}

# Funzione per ottenere il modello SpaCy in base alla lingua
def get_spacy_model(language):
    model_name = LANGUAGE_MODELS.get(language.lower(), "en_core_web_sm") # Inglese come default
    if model_name in loaded_models:
        return loaded_models[model_name]
    try:
        nlp = spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    loaded_models[model_name] = nlp
    return nlp

# Funzione per verificare se il campo "entities" è valido
def is_entities_valid(entities):
    if not isinstance(entities, dict): 
        return False
    if not entities: 
        return False
    for entity_type, entity_list in entities.items():
        if not isinstance(entity_list, list):
            return False
        if not all(isinstance(entity, dict) for entity in entity_list):
            return False
        if not all("name" in entity for entity in entity_list):
            return False
        if not entity_list:
            return False
    return True

# Funzione per estrarre entità dal testo utilizzando SpaCy
def extract_entities(text, language="en"):
    try:
        # Carica il modello SpaCy per la lingua specificata
        nlp = get_spacy_model(language)
        # Processa il testo con il modello SpaCy
        doc = nlp(text)
        entities = {
            "persons": [],
            "organizations": [],
            "locations": []
        }
        for ent in doc.ents: 
            if ent.label_ == "PERSON": # Identifica le entità di tipo PERSON
                # Aggiunge solo il nome in maiuscolo
                entities["persons"].append(ent.text.title())
            elif ent.label_ == "ORG": # Identifica le entità di tipo ORG
                # Aggiunge solo il nome in maiuscolo
                entities["organizations"].append(ent.text.title())
            elif ent.label_ in ["GPE", "LOC"]: # Identifica le entità di tipo GPE (Geopolitical Entity) o LOC (Location)
                # Aggiunge solo il nome in maiuscolo
                entities["locations"].append(ent.text.title())
        return entities
    except Exception as e:
        print(f"Errore durante l'estrazione delle entità: {e}")
        return {
            "persons": [],
            "organizations": [],
            "locations": []
        }

# Connessione a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Seleziona documenti che hanno i campi "entities" e "text_processed" ma non "entities_processed"
query = {
    "entities": {"$exists": True}, 
    "text_processed": {"$exists": True},
    "entities_processed": {"$exists": False}
}
documents = list(collection.find(query))

# Itera su ogni documento recuperato e aggiungi il nuovo campo "entities_processed"
for doc in tqdm(documents, desc="Processing documents"):
    if is_entities_valid(doc["entities"]): # Verifica se il campo "entities" è valido
        # Se il campo "entities" è valido, "entities_processed" è una copia con solo i nomi in maiuscolo (title())
        entities_processed = {
            entity_type: [entity["name"].title() for entity in entity_list]
            for entity_type, entity_list in doc["entities"].items()
        }
        # Aggiunge al documento il nuovo campo "entities_processed"
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"entities_processed": entities_processed}}
        )
    else:
        # Se il campo "entities" non è valido, esegue l'estrazione delle entità dal testo processato
        if "text" in doc:
            language = doc.get("language", "en")
            entities_processed = extract_entities(doc.get("text_processed"), language)
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"entities_processed": entities_processed}}
            )
        else:
            # In caso di problemi, imposta "entities_processed" come un dizionario vuoto con liste vuote
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"entities_processed": {
                    "persons": [],
                    "organizations": [],
                    "locations": []
                }}}
            )

# Stampa un messaggio di completamento con il numero di documenti processati
print(f"Campo 'entities_processed' popolato in {len(documents)} documenti.")
