# build_index.py

import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
import pdfplumber
from openai import OpenAI

DATA_PATH     = "data/pdfs"
INDEX_FILE    = "faiss.index"
META_FILE     = "faiss_meta.pkl"
EMB_MODEL     = "text-embedding-3-small"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 100

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_pdf_text(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def get_chunks(text: str) -> List[str]:
    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(L, start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

# 1) Fragmentar y recolectar
texts, metas = [], []
for fname in os.listdir(DATA_PATH):
    if fname.lower().endswith(".pdf"):
        full = get_pdf_text(os.path.join(DATA_PATH, fname))
        for chunk in get_chunks(full):
            texts.append(chunk)
            metas.append({"text": chunk, "source": fname})

# 2) Embeddings en lotes
emb_list = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=batch)
    for e in resp.data:
        emb_list.append(e.embedding)

# 3) Indexar con FAISS
emb_matrix = np.array(emb_list, dtype="float32")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

# 4) Persistir
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(metas, f)
