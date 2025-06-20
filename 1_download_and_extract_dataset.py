import os
import requests
import zipfile
import shutil

GITHUB_URL = "https://github.com/Webhose/free-news-datasets/archive/refs/heads/master.zip"
EXTRACT_DIR = "News_Datasets"
MAIN_ZIP = "master.zip"

def download_and_extract():
    """
    Scarica e prepara il dataset di news dal repository GitHub di Webhose.
    """
    print("Scaricamento del dataset in corso...")
    # 1. Scarica il file zip principale dal repository GitHub
    r = requests.get(GITHUB_URL)
    r.raise_for_status()
    with open(MAIN_ZIP, "wb") as f:
        f.write(r.content)
    print("Download completato.")

    # 2. Estrae tutto il contenuto dello zip scaricato
    print("Estrazione dell'archivio principale...")
    with zipfile.ZipFile(MAIN_ZIP) as z:
        z.extractall()
    print("Estrazione completata.")

    # 3. Sposta la cartella News_Datasets nella directory di lavoro corrente
    src = "free-news-datasets-master/News_Datasets"
    if os.path.exists(EXTRACT_DIR):
        print(f"La cartella '{EXTRACT_DIR}' esiste già.")
    else:
        os.rename(src, EXTRACT_DIR)
        print(f"Cartella '{EXTRACT_DIR}' spostata nella directory di lavoro.")

    # 4. Elimina la cartella del repository estratto (non serve più)
    if os.path.exists("free-news-datasets-master"):
        shutil.rmtree("free-news-datasets-master")
        print("Cartella temporanea del repository eliminata.")

    # 5. Elimina il file zip principale scaricato
    if os.path.exists(MAIN_ZIP):
        os.remove(MAIN_ZIP)
        print("Archivio zip principale eliminato.")

    # 6. Estrae tutti i file zip presenti dentro News_Datasets e li elimina dopo l'estrazione
    print("Estrazione di tutti i file zip presenti nella cartella News_Datasets...")
    for fname in os.listdir(EXTRACT_DIR):
        if fname.endswith(".zip"):
            zippath = os.path.join(EXTRACT_DIR, fname)
            with zipfile.ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            os.remove(zippath)
    print("Estrazione di tutti i file zip completata.")

if __name__ == "__main__":
    download_and_extract()