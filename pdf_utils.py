"""
pdf_utils.py
-------------
Funciones para descargar y extraer texto de PDFs desde Google Drive.
Usa la misma service account y caché en disco.
"""

import io
import os
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pdfplumber

# ======== CONFIGURACIÓN ========
SERVICE_ACCOUNT_FILE = "h360chatbot-sheets.json"  # tu JSON de credenciales
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cargamos credenciales UNA sola vez
CREDS = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
DRIVE = build("drive", "v3", credentials=CREDS, cache_discovery=False)


# ======== FUNCIONES PRINCIPALES ========
def download_pdf_bytes(file_id: str) -> bytes:
    """
    Descarga el PDF desde Drive y devuelve los bytes.
    """
    request = DRIVE.files().get_media(fileId=file_id)
    return request.execute()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extrae y devuelve todo el texto de los bytes PDF.
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def get_pdf_text(file_id: str, force_refresh: bool = False) -> str:
    """
    Devuelve el texto completo de un PDF, usando caché en disk.
    - Si existe cache (cache/<file_id>.txt) y force_refresh=False, lo usa.
    - Si no, descarga y extrae de nuevo.
    """
    cache_path = os.path.join(CACHE_DIR, f"{file_id}.txt")
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    pdf_bytes = download_pdf_bytes(file_id)
    full_text = extract_text_from_pdf_bytes(pdf_bytes)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text
