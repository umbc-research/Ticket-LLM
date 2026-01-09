import pandas as pd
import os
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
CSV_FILE = 'cleaned_tickets.csv'
WIKI_DIR = 'wiki_content'
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
    
    # We use different splitters for tickets vs manuals for better context retention
    ticket_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    wiki_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)

    # 1. Process Tickets
    if os.path.exists(CSV_FILE):
        print(f"Indexing tickets from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        raw_ticket_docs = []
        for idx, row in df.iterrows():
            text = f"TICKET ID: {row['TicketID']}\nSUBJECT: {row['SubjectNoHTML']}\nCONTENT: {row['TransactionContent']}"
            raw_ticket_docs.append(Document(page_content=text, metadata={"source": "ticket"}))
        all_docs.extend(ticket_splitter.split_documents(raw_ticket_docs))

    # 2. Process Wiki Content
    if os.path.exists(WIKI_DIR):
        print(f"Indexing Wiki pages from {WIKI_DIR}...")
        raw_wiki_docs = []
        for filename in os.listdir(WIKI_DIR):
            if filename.endswith(".txt"):
                with open(os.path.join(WIKI_DIR, filename), 'r') as f:
                    content = f.read()
                    raw_wiki_docs.append(Document(page_content=content, metadata={"source": "wiki", "title": filename}))
        all_docs.extend(wiki_splitter.split_documents(raw_wiki_docs))

    print(f"Indexing {len(all_docs)} total chunks...")
    db = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=DB_DIR)
    print("Database built successfully!")
    return db

def ask_assistant(db, question):
    # Retrieve top 15 relevant chunks
    results = db.similarity_search(question, k=15)
    
    # Separate context for the prompt
    context_text = ""
    for res in results:
        if res.metadata.get("source") == "wiki":
            context_text += f"\n[OFFICIAL WIKI]:\n{res.page_content}\n---"
        else:
            context_text += f"\n[PAST TICKET]:\n{res.page_content}\n---"

    prompt = f"""
    You are the UMBC HPC Support Assistant. Answer the USER QUESTION using the provided context.

    RULES:
    1. If information is found in an [OFFICIAL WIKI] block, treat it as the primary instruction. 
    2. If a [WIKI] block contains a 'SOURCE URL', provide that URL to the user.
    3. Use [PAST TICKET] info for real-world examples or if the Wiki is silent.
    4. If information is conflicting, prioritize more recent ticket dates or the official Wiki.
    5. If you don't know the answer, suggest contacting HPC support at UMBC.

    CONTEXT:
    {context_text}

    USER QUESTION:
    {question}

    ANSWER:"""

    print("\n--- 70B is thinking ---")
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    print(f"\n[ASSISTANT]:\n{response['response']}\n")

if __name__ == "__main__":
    vector_db = build_or_load_db()
    while True:
        query = input("Ask the HPC Assistant (or type 'exit'): ")
        if query.lower() == 'exit': break
        if query.strip(): ask_assistant(vector_db, query)
