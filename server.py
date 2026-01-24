import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import os

# --- CONFIG ---
DB_PATH = "chroma_db" 

app = FastAPI()

print("🔌 Loading Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"🔌 Connecting to Database at '{DB_PATH}'...")
if not os.path.exists(DB_PATH):
    print(f"❌ CRITICAL ERROR: Database folder '{DB_PATH}' not found! Run ingest.py.")
else:
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print("✅ Database Connected.")

print("🔌 Loading Reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Server Ready.")

class QueryRequest(BaseModel):
    query: str

class CalcRequest(BaseModel):
    dose: float
    area: float

# --- TOOLS ---

@app.post("/search")
def search_tool(req: QueryRequest):
    print(f"🔎 Searching: {req.query}")
    
    # K=25 allows deep retrieval for specific policy clauses
    docs = vector_db.similarity_search(req.query, k=25)
    
    if not docs: 
        return {"results": [], "metadatas": []}

    # Re-Rank
    pairs = [[req.query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    # Sort by Score
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    valid_results = []
    valid_metas = []
    
    # Top 5 is usually enough for policy context
    for score, doc in scored_docs[:5]:
        if score > -10.0: # Soft threshold
            valid_results.append(doc.page_content)
            valid_metas.append(doc.metadata)
            
    return {
        "results": valid_results,
        "metadatas": valid_metas
    }

@app.post("/calculate")
def calculate_tool(req: CalcRequest):
    # Kept generic: dose=num1, area=num2. Can be used for "Fine Calculation" if needed.
    total = req.dose * req.area
    return {"total": total, "msg": f"CALCULATED: {total}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)