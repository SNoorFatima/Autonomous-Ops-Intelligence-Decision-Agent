from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

KB_DIR = Path("kb_docs")
DB_DIR = Path("rag_db")
COLLECTION = "abida_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks

def main():
    DB_DIR.mkdir(exist_ok=True, parents=True)

    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # rebuild cleanly
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    embedder = SentenceTransformer(EMBED_MODEL)

    doc_id = 0
    for fp in KB_DIR.glob("**/*"):
        if fp.suffix.lower() not in [".txt"]:
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        chunks = chunk_text(text)
        vecs = embedder.encode(chunks, normalize_embeddings=True).tolist()

        ids, metas = [], []
        for idx, ch in enumerate(chunks):
            doc_id += 1
            ids.append(f"chunk_{doc_id}")
            metas.append({"source": fp.name, "chunk": idx})

        col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
        print(f"Indexed {fp.name}: {len(chunks)} chunks")

    print("✅ KB built in rag_db/")

if __name__ == "__main__":
    main()