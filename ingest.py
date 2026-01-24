import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/pdfs"   # Matches your folder structure
DB_PATH = "chroma_db"     # matches server.py

def create_vector_db():
    print(f"🚜 Starting Data Ingestion from: {DATA_PATH}")

    # 1. Find Files
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    if not pdf_files:
        print(f"❌ ERROR: No .pdf files found in '{DATA_PATH}'")
        return

    print(f"   -> Found {len(pdf_files)} PDFs. Loading...")

    # 2. Load Text
    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"   - Loaded: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"   ⚠️ Error loading {pdf_path}: {e}")

    if not documents:
        print("❌ ERROR: No text extracted. Check your PDFs.")
        return

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} text chunks.")

    # 4. Create Database
    print("🧠 Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Fresh start
        
    print("💾 Saving Database...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"✅ SUCCESS: Database ready in '{DB_PATH}'!")

if __name__ == "__main__":
    create_vector_db()