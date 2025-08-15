# main.py – motor QA intacto + re-phraser empático + admin/status 🚦

import os
import re
import sys
import json
import unicodedata
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # en Render no hay faiss-gpu
    import faiss_cpu as faiss  # type: ignore

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
from gspread.client import Client
from gspread.exceptions import WorksheetNotFound
from openai import OpenAI

app = FastAPI()

# ───── Configuración Azure Translator ───────────────────────────────────
AZURE_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
if not (AZURE_ENDPOINT and AZURE_KEY and AZURE_REGION):
    sys.exit(
        "❌ Define AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_KEY y AZURE_TRANSLATOR_REGION"
    )


def azure_translate(text: str, to: str) -> Tuple[str, str]:
    """
    Traduce `text` al idioma `to` y devuelve (texto_traducido, idioma_detectado).
    """
    url = f"{AZURE_ENDPOINT}/translate?api-version=3.0&to={to}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_REGION,
        "Content-Type": "application/json",
    }
    resp = requests.post(url,
                         headers=headers,
                         json=[{
                             "text": text
                         }],
                         timeout=20)
    resp.raise_for_status()
    j = resp.json()[0]
    return j["translations"][0]["text"], j["detectedLanguage"]["language"]


# ───── OpenAI ───────────────────────────────────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    sys.exit("❌ Falta la variable de entorno OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_KEY)

# ───── Prompts ──────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = "Eres un modelo experto en asistencia hotelera y debes seguir las directrices internas."

STYLE_PROMPT = """
Eres el asistente virtual del hotel. Transforma la «Respuesta Base» en un mensaje humano, empático y útil:

1. Contextualiza emoción e intención. Usa tono alegre (✔), empático (✘) o informativo (ℹ️).
2. Lenguaje cercano y profesional — sin tecnicismos ni saludos repetidos; no menciones ubicación.
3. Humor discreto solo si encaja.
4. Esta respuesta es parte de una conversación ya iniciada, por lo que NO debe incluir mensajes de bienvenida ni saludos iniciales de ningún tipo. Comienza directamente con la información solicitada. Es decir nunca uses "Hola" o "buenos dias" ni nada de ese estilo. ESTAS PROHIBIDO DE USAR SALUDOS INICIALES AL INICIO DE LAS RESPUESTAS NO DEBES USARL ESOS TERMINOS EN NINGUN MENSAJE JAMAS
5. no repitas las preguntas del ususario
6. no uses siempre el inicio de "claro" y manten el uso de expresiones introductorias como "Claro", "entiendo", "perfecto", etc. a un 40% de las respuestas

‼️ No inventes datos ni cambies lo factual.
⚠️ Mantén siempre el mismo idioma de la «Respuesta Base».

---
RESPUESTA BASE:
{answer}
---
RESPUESTA CONCIERGE:
""".strip()


# ───── Utilidades texto ─────────────────────────────────────────────────
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def safe(val: Any) -> str:
    return str(val).strip() if val else ""


# ───── Google Sheets – credenciales desde variable de entorno ───────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def load_gcp_creds() -> Credentials:
    """
    Lee el JSON de la service-account directamente desde la variable
    de entorno GCP_CREDENTIALS_JSON y lo convierte en Credentials.
    """
    raw = os.getenv("GCP_CREDENTIALS_JSON")  #  ← NOMBRE DEFINITIVO
    if not raw:
        sys.exit("❌ Falta la variable de entorno GCP_CREDENTIALS_JSON")
    info = json.loads(raw)
    return Credentials.from_service_account_info(info, scopes=SCOPES)


creds = load_gcp_creds()
gs = Client(auth=creds)

# ───── IDs de Sheets ────────────────────────────────────────────────────
OPERATIVAS_ID = "1mG6qseyNI5yQaTrETPbCDha9DXyIQqQJzsmSS1fxFN4"
COMERCIALES_ID = "1UQo2Prd7nk9XA7YwOQOMK2__Q5TJVMMsYpCnjYJHayc"
RESERVAS_ID = "1zvhcb073f0iT16JTMAdlpBUhvIhsEzxOFvK6o8LUJ3Y"

# ───── Estado global del índice (se rellena en rebuild) ─────────────────
QA_ROWS: List[Dict[str, Any]] = []
CORPUS: List[str] = []
index = None  # type: ignore
LAST_REBUILD_AT: Optional[str] = None
APP_VERSION = os.getenv("APP_VERSION", "dev")

# ───── Token admin para endpoints protegidos ────────────────────────────
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    sys.exit("❌ Falta la variable de entorno ADMIN_TOKEN")


def header_row(sh):
    for i, row in enumerate(sh.get_all_values()):
        cells = [safe(c) for c in row if safe(c)]
        if len(cells) >= 2 and len(cells) == len(set(cells)):
            return i
    return 0


def read_records(sh):
    vals = sh.get_all_values()
    if not vals:
        return
    h = header_row(sh)
    heads = [safe(c).lower() for c in vals[h]]
    idx = {name: i for i, name in enumerate(heads)}
    for row in vals[h + 1:]:
        if not any(safe(c) for c in row):
            continue
        yield {
            "hotel":
            safe(row[idx.get("hotel", -1)]),
            "room":
            safe(row[idx.get("habitacion", -1)]),
            "a":
            safe(row[idx.get("respuesta", -1)]),
            "media": [{
                "type": safe(row[idx.get("media_type", -1)]),
                "url": safe(row[idx.get("media_url", -1)])
            }] if safe(row[idx.get("media_type", -1)])
            and safe(row[idx.get("media_url", -1)]) else [],
        }


def load_corpus() -> List[Dict[str, Any]]:
    sheets = [
        gs.open_by_key(OPERATIVAS_ID).sheet1,
        gs.open_by_key(COMERCIALES_ID).sheet1,
    ]
    for tab in ("Capacidad", "Precio", "Disponibilidad"):
        try:
            sheets.append(gs.open_by_key(RESERVAS_ID).worksheet(tab))
        except WorksheetNotFound:
            print(f"[aviso] Hoja '{tab}' omitida.")
    corpus = []
    for sh in sheets:
        corpus.extend(list(read_records(sh)))
    return [r for r in corpus if r["a"]]


# ───── Rebuild del índice (admin) ───────────────────────────────────────
def rebuild_index() -> Dict[str, Any]:
    global QA_ROWS, CORPUS, index, LAST_REBUILD_AT

    print("[admin] Rebuild iniciado…")
    QA_ROWS = load_corpus()
    if not QA_ROWS:
        raise RuntimeError("Sin filas válidas en Sheets")

    CORPUS = [normalize(r["a"]) for r in QA_ROWS]
    print(f"[admin] Generando {len(CORPUS)} embeddings…")
    embs = openai.embeddings.create(model="text-embedding-3-small",
                                    input=CORPUS).data
    mat = np.array([e.embedding for e in embs], dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)  # type: ignore

    LAST_REBUILD_AT = datetime.now(timezone.utc).isoformat()
    print("[admin] Rebuild completado")
    return {
        "total_registros": len(QA_ROWS),
        "dimensiones": int(mat.shape[1]),
    }


# ───── Startup: construir índice una vez ────────────────────────────────
print("[startup] Cargando registros y construyendo índice…")
rebuild_index()
THRESH = 0.45


def retrieve_answer(q: str, hotel: Optional[str],
                    room: Optional[str]) -> Tuple[str, List[Dict[str, str]]]:
    subset = QA_ROWS
    if hotel:
        subset = [r for r in subset if r["hotel"].lower() == hotel.lower()]
    if room:
        subset = [r for r in subset if r["room"].lower() == room.lower()]
    if not subset:
        return "Lo siento, no dispongo de esa información.", []
    cand = [normalize(r["a"]) for r in subset]
    embs2 = openai.embeddings.create(model="text-embedding-3-small",
                                     input=[normalize(q)] + cand).data
    qv = np.array(embs2[0].embedding, dtype="float32")[None, :]
    dv = np.array([e.embedding for e in embs2[1:]], dtype="float32")
    faiss.normalize_L2(qv)
    faiss.normalize_L2(dv)  # type: ignore
    sims = (qv @ dv.T)[0]
    i = int(np.argmax(sims))
    if sims[i] < THRESH:
        return "Lo siento, no dispongo de esa información.", []
    hit = subset[i]
    return hit["a"], hit["media"]


def restyle(answer: str) -> str:
    prompt = STYLE_PROMPT.format(answer=answer)
    rsp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": BASE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.25,
        max_tokens=300,
    ).choices[0].message.content.strip()
    return rsp


class AskPayload(BaseModel):
    question: str
    hotel: Optional[str] = None
    habitacion: Optional[str] = None


class AdminRebuildPayload(BaseModel):
    token: str


@app.post("/ask")
def ask_bot(p: AskPayload):
    q_es, src_lang = azure_translate(p.question, to="es")
    raw, media = retrieve_answer(q_es, p.hotel, p.habitacion)
    text_es = restyle(raw)
    text_out = azure_translate(text_es,
                               to=src_lang)[0] if src_lang != "es" else text_es
    return {"text": text_out, "media": media, "layer": 1}


@app.post("/admin/rebuild")
def admin_rebuild(p: AdminRebuildPayload):
    if p.token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="No autorizado")
    try:
        result = rebuild_index()
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/status")
def admin_status():
    """
    Estado rápido del bot sin reconstruir nada.
    """
    idx_dim = int(getattr(index, "d", 0)) if index is not None else 0
    return {
        "status":
        "ready" if index is not None and len(QA_ROWS) > 0 else "cold",
        "rows": len(QA_ROWS),
        "index_dim": idx_dim,
        "last_rebuild_at": LAST_REBUILD_AT,
        "version": APP_VERSION,
    }


@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok"}


print("Rutas cargadas:",
      [r.path for r in app.routes if isinstance(r, APIRoute)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
