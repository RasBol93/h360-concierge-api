import os
import unicodedata
import difflib
import math
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
from gspread.client import Client
from openai import OpenAI
import pdfplumber

# ───────────────────────────── Configuración OpenAI ─────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────── FastAPI App ───────────────────────────────────────
app = FastAPI()

# ───────────────────────────── Google Sheets ────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds = Credentials.from_service_account_file("h360chatbot-sheets.json",
                                              scopes=SCOPES)
gs_client = Client(auth=creds)

OPERATIVAS_ID = "1mG6qseyNI5yQaTrETPbCDha9DXyIQqQJzsmSS1fxFN4"
COMERCIALES_ID = "1UQo2Prd7nk9XA7YwOQOMK2__Q5TJVMMsYpCnjYJHayc"
RESERVAS_ID = "1zvhcb073f0iT16JTMAdlpBUhvIhsEzxOFvK6o8LUJ3Y"


# ───────────────────────────── Helper: leer PDF local ─────────────────────────────
def get_local_pdf_text(path: str) -> str:
    """
    Extrae texto completo de un PDF local usando pdfplumber.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)


# ───────────────────────────── Utilidades Sheets ─────────────────────────────────
def detect_header_row(sheet):
    vals = sheet.get_all_values()
    for i, row in enumerate(vals):
        clean = [c.strip() for c in row if c.strip()]
        if len(clean) >= 2 and len(clean) == len(set(clean)):
            return i
    return 0


def read_records_manual(sheet):
    vals = sheet.get_all_values()
    if not vals:
        return []
    h = detect_header_row(sheet)
    headers = [c for c in vals[h] if c.strip()]
    idxs = [i for i, c in enumerate(vals[h]) if c.strip()]
    return [
        dict(zip(headers, [row[i] for i in idxs])) for row in vals[h + 1:]
        if any(c.strip() for c in row)
    ]


def normalize(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt.lower())
    return "".join(ch for ch in txt if not unicodedata.combining(ch))


# ───────────────────────────── Carga Layer 1 (Sheets) ────────────────────────────
sh_oper = gs_client.open_by_key(OPERATIVAS_ID).sheet1
sh_comm = gs_client.open_by_key(COMERCIALES_ID).sheet1
spread_res = gs_client.open_by_key(RESERVAS_ID)
sh_cap = spread_res.worksheet("Capacidad")
sh_price = spread_res.worksheet("Precio")
sh_disp = spread_res.worksheet("Disponibilidad")

preg_oper = read_records_manual(sh_oper)
preg_comm = read_records_manual(sh_comm)
reserv_cap = read_records_manual(sh_cap)
reserv_price = read_records_manual(sh_price)
reserv_disp = read_records_manual(sh_disp)

# ───────────────────────────── RAG Casero: embeddings y FAISS ────────────────────
DATA_PATH = "data/pdfs"
CHUNK_SIZE = 1000
OVERLAP = 200
EMB_MODEL = "text-embedding-3-small"


def get_chunks(text: str) -> List[str]:
    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(L, start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start = end - OVERLAP
    return chunks


pdf_chunks: List[str] = []
pdf_embeddings: List[List[float]] = []

# 1) Leer y fragmentar PDFs locales
for fn in os.listdir(DATA_PATH):
    if fn.lower().endswith(".pdf"):
        full = get_local_pdf_text(os.path.join(DATA_PATH, fn))
        for chunk in get_chunks(full):
            pdf_chunks.append(chunk)

# 2) Calcular embeddings en batches de 100
for i in range(0, len(pdf_chunks), 100):
    batch = pdf_chunks[i:i + 100]
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=batch)
    for item in resp.data:
        pdf_embeddings.append(item.embedding)


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-8
    nb = math.sqrt(sum(x * x for x in b)) + 1e-8
    return dot / (na * nb)


def rag_search(query: str, k: int = 4) -> List[str]:
    q_resp = openai_client.embeddings.create(model=EMB_MODEL, input=[query])
    q_emb = q_resp.data[0].embedding
    scores = [(cosine_sim(q_emb, emb), idx)
              for idx, emb in enumerate(pdf_embeddings)]
    topk = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return [pdf_chunks[idx] for _, idx in topk]


# ───────────────────────────── Búsqueda en Sheets ──────────────────────────────
def attach_media(rec: Dict) -> List[Dict]:
    t = (rec.get("media_type") or "").strip()
    u = (rec.get("media_url") or "").strip()
    return [{"type": t, "url": u}] if t and u else []


def find_in_sheets(q: str) -> Optional[tuple]:
    toks = normalize(q).split()
    for ds in (preg_oper, preg_comm, reserv_cap + reserv_price + reserv_disp):
        for r in ds:
            hay = normalize(" ".join(str(v) for v in r.values()))
            if all(t in hay for t in toks):
                return (r.get("respuesta", ""), attach_media(r))
    return None


# ───────────────────────────── GPT para pulir respuesta ────────────────────────
SYSTEM = (
    "Eres el asistente virtual del Hotel H360. "
    "Responde siempre en español, cordial y usa SOLO el contexto proporcionado."
)


def get_best_answer(q: str,
                    hotel: Optional[str] = None,
                    habitacion: Optional[str] = None) -> Dict:
    # 1) Layer 1: Sheets
    hit = find_in_sheets(q)
    if hit:
        raw, media = hit
        layer = 1
    else:
        # 2) Layer 2: RAG local
        chunks = rag_search(q, k=4)
        raw = "\n\n".join(
            chunks) if chunks else "Lo siento, no dispongo de esa información."
        media = []
        layer = 2

    try:
        chat = openai_client.chat.completions.create(model="gpt-3.5-turbo",
                                                     messages=[{
                                                         "role":
                                                         "system",
                                                         "content":
                                                         SYSTEM
                                                     }, {
                                                         "role":
                                                         "assistant",
                                                         "content":
                                                         f"Contexto:\n{raw}"
                                                     }, {
                                                         "role": "user",
                                                         "content": q
                                                     }],
                                                     temperature=0.3,
                                                     max_tokens=256)
        out = chat.choices[0].message.content.strip()
    except Exception:
        out = raw

    return {"text": out, "layer": layer, "media": media}


# ───────────────────────────── Endpoints ──────────────────────────────────────
class AskPayload(BaseModel):
    question: str
    hotel: Optional[str] = None
    habitacion: Optional[str] = None


@app.post("/ask")
def ask_bot(p: AskPayload):
    return get_best_answer(p.question, p.hotel, p.habitacion)


@app.get("/ask_get")
def ask_get(question: str,
            hotel: Optional[str] = None,
            habitacion: Optional[str] = None):
    return get_best_answer(question, hotel, habitacion)


@app.get("/")
def root():
    return {"status": "ok", "message": "H360ChatBot API is alive"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)),
                reload=True)
