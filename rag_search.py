import pandas as pd
import os
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
CSV_FILE = 'cleaned_tickets.csv'
WIKI_FILE = 'wiki_map.txt'  # New high-priority source
DB_DIR = './ticket_db'
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama-70b-hpc"

def build_or_load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_DIR):
        print(f"--- Loading existing database from {DB_DIR} ---")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print(f"--- Creating new database ---")
    all_docs = []

    # 1. Process Ticket Data (Requires Splitting)
    if os.path.exists(CSV_FILE):
        print(f"Processing tickets from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        raw_ticket_docs = []
        for idx, row in df.iterrows():
            text_content = f"TICKET ID: {row['TicketID']}\nSUBJECT: {row['SubjectNoHTML']}\nCONTENT: {row['TransactionContent']}"
            metadata = {"source": "ticket", "id": str(row['TicketID'])}
            raw_ticket_docs.append(Document(page_content=text_content, metadata=metadata))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        all_docs.extend(text_splitter.split_documents(raw_ticket_docs))

    # 2. Process Wiki Map (No Splitting - keep links whole)
    if os.path.exists(WIKI_FILE):
        print(f"Processing Wiki Map from {WIKI_FILE}...")
        with open(WIKI_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    # Prefix with 'WIKI LINK' so the LLM knows this is official
                    wiki_content = f"WIKI LINK / OFFICIAL DOCUMENTATION: {line.strip()}"
                    all_docs.append(Document(page_content=wiki_content, metadata={"source": "wiki"}))

    print(f"Indexing {len(all_docs)} total chunks...")
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Database built successfully!")
    return db

def ask_assistant(db, question):
    print(f"\n[USER QUESTION]: {question}")

    # Search for top 15 results (Wiki links + Tickets)
    results = db.similarity_search(question, k=15)
    context_text = "\n\n---\n\n".join([res.page_content for res in results])

    prompt = f"""
    You are an HPC Sysadmin Assistant at UMBC. 
    Your goal is to provide accurate technical support using the context below.

    STRICT GUIDELINES:
    1. If the context contains a 'WIKI LINK', provide that link to the user as the primary source of truth.
    2. Use the 'TICKET ID' entries to provide real-world troubleshooting steps from past issues.
    3. If the context doesn't have the answer, provide general HPC best practices (sbatch, module loads, etc.).
    4. Be professional, technical, and include direct links whenever they are found in the context.

    CONTEXT (TICKETS & WIKI):
    {context_text}

    USER QUESTION:
    {question}

    ANSWER:"""

    print("--- 70B is thinking ---")
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)

    print("\n[ASSISTANT RESPONSE]:")
    print(response['response'])
    print("\n--------------------------")

if __name__ == "__main__":
    vector_db = build_or_load_db()

    while True:
        query = input("\nAsk the HPC Assistant (or type 'exit'): ")
        if query.lower() == 'exit': break
        if query.strip():
            ask_assistant(vector_db, query)
