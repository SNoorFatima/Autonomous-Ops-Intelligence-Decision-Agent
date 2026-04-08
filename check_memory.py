
import chromadb
from chromadb.config import Settings
import os

DB_DIR = "rag_db"
HISTORY_COLLECTION_NAME = "abida_history"

def check_memory():
    if not os.path.exists(DB_DIR):
        print(f"Directory {DB_DIR} does not exist.")
        return

    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        ),
        tenant="default_tenant",
        database="default_database"
    )
    
    try:
        col = client.get_collection(HISTORY_COLLECTION_NAME)
        count = col.count()
        print(f"Success: History collection found.")
        print(f"Total entries in memory: {count}")
        
        if count > 0:
            res = col.get(limit=5, include=["documents", "metadatas"])
            print("\nRecent memory entries:")
            for i in range(len(res["ids"])):
                print(f"ID: {res['ids'][i]}")
                print(f"Type: {res['metadatas'][i].get('type')}")
                print(f"Timestamp: {res['metadatas'][i].get('timestamp')}")
                print(f"Summary: {res['documents'][i][:100]}...")
                print("-" * 20)
        else:
            print("The memory collection is empty.")
            
    except Exception as e:
        print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    check_memory()
