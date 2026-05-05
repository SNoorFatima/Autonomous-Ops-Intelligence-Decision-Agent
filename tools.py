import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# -----------------------------
# RAG (Vector DB) Configuration
# -----------------------------
DB_DIR = "rag_db"
COLLECTION_NAME = "abida_knowledge"
HISTORY_COLLECTION_NAME = "abida_history"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

_client = None


def _get_client():
    global _client
    if _client is None:
        chroma_host = os.environ.get("CHROMA_HOST")
        if chroma_host:
            print(f"[Chroma] Connecting to remote ChromaDB at {chroma_host}:8000")
            _client = chromadb.HttpClient(
                host=chroma_host,
                port=8000,
                settings=Settings(
                    anonymized_telemetry=False
                ),
                tenant="default_tenant",
                database="default_database"
            )
        else:
            print(f"[Chroma] Using local PersistentClient at {DB_DIR}")
            _client = chromadb.PersistentClient(
                path=DB_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                ),
                tenant="default_tenant",
                database="default_database"
            )
    return _client


def _get_collection():
    return _get_client().get_collection(COLLECTION_NAME)

def _get_history_collection():
    # Use get_or_create to ensure the history collection exists
    return _get_client().get_or_create_collection(HISTORY_COLLECTION_NAME)


# -----------------------------
# Tool 1: RAG Grounding Search
# -----------------------------
class GroundingInput(BaseModel):
    query: str = Field(..., description="Question to search in PDFs/SOP knowledge base.")
    top_k: int = Field(2, ge=1, le=5, description="Number of evidence chunks to retrieve.")

@tool("grounding_search", args_schema=GroundingInput)
def grounding_search(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Search the PDF knowledge base and return top evidence chunks with sources.
    Use for SOP/policy/definitions/best-practices questions.
    """
    print(f"[Tool: RAG] Searching knowledge base for: '{query}'...")
    col = _get_collection()
    embedder = _get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        truncated_doc = doc[:1000] + "..." if len(doc) > 1000 else doc
        hits.append({
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", None),
            "text": truncated_doc,
            "score": float(1 - dist)
        })
    print(f"[Tool: RAG] Found {len(hits)} relevant chunks.")
    return hits

# -----------------------------
# Tool 2: KPI Analysis on CSV
# -----------------------------
class KPIInput(BaseModel):
    csv_path: str = Field(..., description="Path to cleaned CSV file, e.g. data/global_superstore_clean.csv")

@tool("analyze_kpis", args_schema=KPIInput)
def analyze_kpis(csv_path: str) -> Dict[str, Any]:
    """
    Analyze cleaned superstore-style CSV and return KPI summary for ops decisions.
    Expects (at least): order_date, ship_date, sales, profit, region, category, ship_mode
    Optional: discount, shipping_cost, sub_category, quantity
    """
    print(f"[Tool: Analytics] Analyzing CSV: {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"[Tool: Analytics] ERROR: File not found: {csv_path}")
        return {"error": f"File not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    print(f"[Tool: Analytics] Data loaded. Processing {len(df)} rows...")

    # Normalize column names for case-insensitivity
    df.columns = [c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Required for delay analysis
    core_required = ["order_date", "ship_date", "region", "category", "ship_mode"]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        print(f"[Tool: Analytics] ERROR: Missing core columns: {missing}")
        return {"error": f"Missing core columns: {missing}", "available_columns": list(df.columns)}

    # Optional but desired
    has_profit = "profit" in df.columns
    has_sales = "sales" in df.columns

    # Dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Ship delay
    df["ship_delay_days"] = (df["ship_date"] - df["order_date"]).dt.days
    
    initial_count = len(df)
    drop_cols = core_required + ["ship_delay_days"]
    if has_profit: drop_cols.append("profit")
    if has_sales: drop_cols.append("sales")

    df = df.dropna(subset=drop_cols)
    df = df[df["ship_delay_days"] >= 0]
    final_count = len(df)
    
    print(f"[Tool: Analytics] Cleaned data: {final_count} rows remains (dropped {initial_count - final_count} due to missing/invalid dates).")

    if final_count == 0:
        return {
            "error": "Dataset is empty after removing rows with missing dates (ship_date) or invalid delays.",
            "suggestion": "Check the CSV source quality."
        }

    res = {
        "rows_used": int(len(df)),
        "avg_ship_delay_days": round(float(df["ship_delay_days"].mean()), 2),
        "p90_ship_delay_days": round(float(np.percentile(df["ship_delay_days"], 90)), 2),
        "top_delay_regions_avg_days": df.groupby("region")["ship_delay_days"].mean().sort_values(ascending=False).head(5).round(2).to_dict(),
        "delay_by_ship_mode_avg_days": df.groupby("ship_mode")["ship_delay_days"].mean().sort_values(ascending=False).round(2).to_dict(),
        "flags": []
    }

    if has_sales:
        res["total_sales"] = round(float(df["sales"].sum()), 2)
    
    if has_profit and has_sales:
        total_sales = df["sales"].sum()
        total_profit = df["profit"].sum()
        res["total_profit"] = round(float(total_profit), 2)
        res["profit_margin_%"] = round(float((total_profit / total_sales) * 100), 2) if total_sales else 0.0
        
        if res["profit_margin_%"] <= 5:
            res["flags"].append("Profit margin is low (≤ 5%).")

    if res["avg_ship_delay_days"] >= 5:
        res["flags"].append("Shipping delays are high (avg ≥ 5 days).")

    print("[Tool: Analytics] Analysis complete.")
    return res

# -----------------------------
# Tool 3: Store Analysis Memory
# -----------------------------
class StoreMemoryInput(BaseModel):
    summary: str = Field(..., description="A concise executive summary of the analysis findings and recommendations.")
    analysis_type: str = Field("operations_review", description="Type of analysis, e.g., shipping_delay, profit_review.")

@tool("store_analysis_result", args_schema=StoreMemoryInput)
def store_analysis_result(summary: str, analysis_type: str = "operations_review") -> str:
    """
    Save the final analysis summary to long-term memory for future comparison.
    Call this at the very end of your workflow once you have a final conclusion.
    """
    print(f"[Tool: Memory] Storing analysis in history...")
    col = _get_history_collection()
    embedder = _get_embedder()
    
    # Simple ID generation based on time
    import time
    doc_id = f"mem_{int(time.time())}"
    
    vec = embedder.encode([summary], normalize_embeddings=True).tolist()[0]
    
    col.add(
        ids=[doc_id],
        embeddings=[vec],
        documents=[summary],
        metadatas=[{"type": analysis_type, "timestamp": time.ctime()}]
    )
    print(f"[Tool: Memory] Stored successfully as {doc_id}.")
    return f"Successfully stored analysis in history as {doc_id}."

# -----------------------------
# Tool 4: Retrieve Past Memory
# -----------------------------
class RetrieveMemoryInput(BaseModel):
    query: str = Field(..., description="Query to search for past analysis results or trends.")
    top_k: int = Field(2, ge=1, le=5, description="Number of past records to retrieve.")

@tool("retrieve_past_analyses", args_schema=RetrieveMemoryInput)
def retrieve_past_analyses(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Search historical analysis results to find trends or previous findings.
    Use this to compare current data with 'how things were before'.
    """
    print(f"[Tool: Memory] Retrieving history for: '{query}'...")
    col = _get_history_collection()
    
    # Check if empty
    if col.count() == 0:
        print("[Tool: Memory] History is currently empty.")
        return []

    embedder = _get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "timestamp": meta.get("timestamp"),
            "type": meta.get("type"),
            "summary": doc,
            "relevance_score": float(1 - dist)
        })
    print(f"[Tool: Memory] Found {len(hits)} relevant memory snapshots.")
    return hits

