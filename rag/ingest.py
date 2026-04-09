import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdfs():
    # Find all PDFs in the base scratch directory
    pdf_files = glob.glob("../*.pdf")
    
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files. Loading...")
    
    documents = []
    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
            print(f"Loaded {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    print(f"Total pages loaded: {len(documents)}")
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total text chunks: {len(chunks)}")
    
    # Embedding
    print("Generating embeddings (this may take a minute based on CPU)...")
    # Using a small, fast model suitable for CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector database locally
    os.makedirs("vector_db", exist_ok=True)
    vectorstore.save_local("vector_db")
    print("Successfully built and saved FAISS vector database to rag/vector_db/")

if __name__ == "__main__":
    ingest_pdfs()
