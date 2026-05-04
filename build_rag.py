import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DATA_FOLDER = "kb_docs"
DB_FOLDER = "rag_db"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

from tools import _get_client
client = _get_client()

collection = client.get_or_create_collection("abida_knowledge")

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

print("Reading PDFs...")

doc_id = 0
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(DATA_FOLDER, file)
        print("Processing:", file)

        text = read_pdf(path)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()

        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id += 1
            ids.append(f"doc_{doc_id}")
            metadatas.append({"source": file})

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

print("✅ RAG database created successfully")