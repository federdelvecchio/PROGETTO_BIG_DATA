@echo off
set LOG_FILE=updating_the_db_log.txt

REM Aggiungi data e ora al file di log
echo [%date% %time%] Inizio esecuzione pipeline >> %LOG_FILE%

REM Esegui gli script in ordine
for %%S in (0_add_new_data.py 4_clean_text.py 5_summarization.py 6_NER.py  7_chunk_and_embed.py) do (
    echo [%date% %time%] Esecuzione %%S >> %LOG_FILE%
    py %%S >> %LOG_FILE% 2>&1
    if errorlevel 1 (
        echo [%date% %time%] Errore durante l'esecuzione di %%S >> %LOG_FILE%
        exit /b 1
    )
)

REM Fine della pipeline
echo [%date% %time%] Pipeline completata con successo >> %LOG_FILE%
echo ================================================== >> %LOG_FILE%
