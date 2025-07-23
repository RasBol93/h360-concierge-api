
# build_index.py

import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
from openai import OpenAI
import pdfplumber

# ───── Configuración ─────
DATA_PATH     = "data/pdfs"
INDEX_FILE    = "faiss.index"
META_FILE     = "faiss_meta.pkl"
EMB_MODEL     = "text-embedding-3-small"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 100

# Inicializa cliente OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_pdf_text(path: str) -> str:
    """
    Extrae texto completo de un PDF usando pdfplumber.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)

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
metas: List[Dict] = []

print(f"[build_index] Procesando PDFs en {DATA_PATH}...")

for fname in os.listdir(DATA_PATH):
    if fname.lower().endswith(".pdf"):
        print(f"[build_index] Procesando {fname}...")
        full_text = get_pdf_text(os.path.join(DATA_PATH, fname))
        chunks = get_chunks(full_text)
        print(f"[build_index] {fname}: {len(chunks)} fragmentos")
        
        for chunk in chunks:
            texts.append(chunk)
            metas.append({"text": chunk, "source": fname})

print(f"[build_index] Total de fragmentos: {len(texts)}")

# ────────────────────────────────────────────────────────────────
# 2) Calcular embeddings por lotes
# ────────────────────────────────────────────────────────────────
print(f"[build_index] Calculando embeddings...")
emb_list: List[List[float]] = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    print(f"[build_index] Procesando lote {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}")
    
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=batch)
    emb_list.extend([item.embedding for item in resp.data])

# ────────────────────────────────────────────────────────────────
# 3) Construir y poblar el índice FAISS
# ────────────────────────────────────────────────────────────────
print(f"[build_index] Construyendo índice FAISS...")
emb_matrix = np.array(emb_list, dtype="float32")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

# ────────────────────────────────────────────────────────────────
# 4) Persistir el índice y la metadata
# ────────────────────────────────────────────────────────────────
print(f"[build_index] Guardando índice y metadata...")
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(metas, f)

print(f"[build_index] ✅ Terminó. Indexados {len(texts)} fragmentos.")
print(f"[build_index] Archivos creados:")
print(f"  - {INDEX_FILE}")
print(f"  - {META_FILE}")
