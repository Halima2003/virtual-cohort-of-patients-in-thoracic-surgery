"""
PHASE 1 — Re-indexation RAG
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import re
import chromadb
import ollama
from docx import Document

DOCX_FILE  = "guidelines_RAG.docx"
COLLECTION = "thoracic_v2"
EMBED_MODEL = "nomic-embed-text"

# ── 1. Lire le docx et découper par sections ────────────────────────────────

doc = Document(DOCX_FILE)

sections = []          # liste de {"title": ..., "text": ..., "index": ...}
current_title = "Introduction"
current_body  = []

for para in doc.paragraphs:
    style = para.style.name
    text  = para.text.strip()
    if not text:
        continue
    if style.startswith("Heading"):
        # Sauvegarder la section précédente si non-vide
        if current_body:
            sections.append({
                "title": current_title,
                "text" : current_title + "\n" + "\n".join(current_body),
                "index": len(sections),
            })
            current_body = []
        current_title = text
    else:
        current_body.append(text)

# Dernière section
if current_body:
    sections.append({
        "title": current_title,
        "text" : current_title + "\n" + "\n".join(current_body),
        "index": len(sections),
    })

print(f"Sections trouvées dans {DOCX_FILE} : {len(sections)}")
for i, s in enumerate(sections):
    words = len(s["text"].split())
    print(f"  [{i:>2}] {s['title'][:60]:<60}  ({words} mots)")

# ── 2. Connexion ChromaDB — recréer la collection proprement ────────────────

client = chromadb.PersistentClient(path="./chroma_db")

# Supprimer l'ancienne v2 si elle existe (pour re-indexer proprement)
try:
    client.delete_collection(COLLECTION)
    print(f"\nAncienne collection '{COLLECTION}' supprimée.")
except Exception:
    pass

collection = client.create_collection(name=COLLECTION)
print(f"Collection '{COLLECTION}' créée.")

# ── 3. Embedding et indexation ───────────────────────────────────────────────

print(f"\nIndexation de {len(sections)} sections…")

for s in sections:
    idx   = s["index"]
    text  = s["text"]
    title = s["title"]

    print(f"  [{idx+1:>2}/{len(sections)}] Embedding : {title[:55]}")

    emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

    collection.add(
        ids       =[f"sec_{idx:03d}"],
        documents =[text],
        embeddings=[emb],
        metadatas =[{
            "title" : title,
            "index" : idx,
            "words" : len(text.split()),
        }],
    )

print(f"\nIndexation terminée — {collection.count()} chunks dans '{COLLECTION}'")

# ── 4. Test de récupération rapide ───────────────────────────────────────────

print("\nTest de récupération (query : 'VEMS DLCO operabilite chirurgie') :")
q_emb = ollama.embeddings(model=EMBED_MODEL, prompt="VEMS DLCO opérabilité chirurgie")["embedding"]
res = collection.query(query_embeddings=[q_emb], n_results=3)
for i, (doc_text, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0])):
    print(f"  [{i+1}] {meta['title']} — {doc_text[:120].replace(chr(10),' ')}…")

print("\nPhase 1 — RAG indexé avec succès.")
