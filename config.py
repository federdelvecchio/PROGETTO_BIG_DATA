import streamlit as st

# Legge le credenziali dal file .streamlit/secrets.toml
MONGO_URI = st.secrets.get("MONGO_URI", "mongodb://localhost:27017/")

# Nome del database e della collezione principale
DB_NAME = "facts_db"
COLLECTION_NAME = "articles"

# Definizione dei campi da estrarre e mappare
FIELDS_MAP = {
    "uuid": ("uuid", "thread.uuid"),
    "url": ("url", "thread.url"),
    "author": ("author",),
    "published": ("published", "thread.published"),
    "title": ("title_full", "thread.title_full"),
    "text": ("text",),
    "language": ("language",),
    "sentiment": ("sentiment",),
    "categories": ("categories",),
    "crawled": ("crawled",),
    "updated": ("updated",),
    "site": ("thread.site_full",),
    "site_type": ("thread.site_type",),
    "country": ("thread.country",),
    "main_image": ("thread.main_image",),
    "domain_rank": ("thread.domain_rank",),
    "domain_rank_updated": ("thread.domain_rank_updated",),
    "site_categories": ("thread.site_categories",),
    "entities": ("entities",)
}