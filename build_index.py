# build_index.py

import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
import openai
from openai import Embedding

from pdf_utils import get_pdf_text

# ───── Configuración ─────
DATA_PATH     = "data/pdfs"
INDEX_FILE    = "faiss.index"
META_FILE     = "faiss_meta.pkl"
EMB_MODEL     = "text-embedding-3-small"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 100

# Inicializa tu API Key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chunks(text: str) -> List[str]:
    """Divide un texto en fragmentos solapados."""
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

# ────────────────────────────────────────────────────────────────
# 1) Leer PDFs, fragmentarlos y acumular metadata
# ────────────────────────────────────────────────────────────────
texts: List[str] = []
metas:  List[Dict] = []

for fname in os.listdir(DATA_PATH):
    if fname.lower().endswith(".pdf"):
        full_text = get_pdf_text(os.path.join(DATA_PATH, fname))
        for chunk in get_chunks(full_text):
            texts.append(chunk)
            metas.append({"text": chunk, "source": fname})

# ────────────────────────────────────────────────────────────────
# 2) Calcular embeddings por lotes
# ────────────────────────────────────────────────────────────────
emb_list: List[List[float]] = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    resp  = Embedding.create(model=EMB_MODEL, input=batch)
    emb_list.extend([d["embedding"] for d in resp["data"]])

# ────────────────────────────────────────────────────────────────
# 3) Construir y poblar el índice FAISS
# ────────────────────────────────────────────────────────────────
emb_matrix = np.array(emb_list, dtype="float32")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

# ────────────────────────────────────────────────────────────────
# 4) Persistir el índice y la metadata
# ────────────────────────────────────────────────────────────────
faiss.write_index(index, INDEX_FILE)     # <- ahora recibe (index, ruta)
with open(META_FILE, "wb") as f:
    pickle.dump(metas, f)

print(f"[build_index] Terminó. Indexados {len(texts)} fragmentos.")
