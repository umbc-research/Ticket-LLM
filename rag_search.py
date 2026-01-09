import pandas as pd
import os
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
CSV_FILE = 'cleaned_tickets.csv'
DB_DIR = './ticket_db'
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama-70b-hpc"

def build_or_load_db():
    # 1. Setup the 'Translator' (Embeddings)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 2. Check if we already have a saved database
    if os.path.exists(DB_DIR):
        print(f"--- Loading existing database from {DB_DIR} ---")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 3. If no database exists, create one
    print(f"--- Creating new database from {CSV_FILE} ---")
    df = pd.read_csv(CSV_FILE)
    
    # We'll process all tickets now that we have a splitter
    print(f"Processing {len(df)} tickets...")
    
    raw_documents = []
    for idx, row in df.iterrows():
        # Combine subject and content for context
        text_content = f"SUBJECT: {row['SubjectNoHTML']}\nCONTENT: {row['TransactionContent']}"
        
        # Keep track of Ticket ID and User in metadata
        metadata = {
            "id": str(row['TicketID']),
            "date": str(row['CreatedDate']),
            "subject": str(row['SubjectNoHTML']),
            "requestor": str(row['Requestor'])
        }        
        raw_documents.append(Document(page_content=text_content, metadata=metadata))

    # 4. Split long tickets into manageable chunks (Fixes the 500 error)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split tickets into {len(documents)} searchable chunks.")

    # 5. Build and save the Vector Store
    print("Indexing... this may take a few minutes for 3,000 tickets.")
    db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("Database built successfully!")
    return db

def ask_assistant(db, question):
    print(f"\n[USER QUESTION]: {question}")
    
    # 1. Search for the 15 most relevant pieces of past tickets
    results = db.similarity_search(question, k=15)
    
    # 2. Extract the text for the LLM context
    context_text = "\n\n---\n\n".join([res.page_content for res in results])
    
    # 3. Craft the prompt for the 70B model
    prompt = f"""
    You are an HPC Sysadmin Assistant at UMBC. 
    Use the following pieces of PAST TICKET HISTORY to answer the USER QUESTION.
    
    PAST TICKET HISTORY:
    {context_text}
    
    USER QUESTION:
    {question}
    
    INSTRUCTIONS:
    - If the history contains a solution, summarize it as a step-by-step guide.
    - Reference specific ticket IDs or past usernames if it helps establish context.
    - If the history is not relevant, provide general HPC troubleshooting steps.
    - Be professional, technical, and concise.
    """

    print("--- 70B is thinking ---")
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    
    print("\n[ASSISTANT RESPONSE]:")
    print(response['response'])
    print("\n--------------------------")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure the 70B model is ready
    # Initialize the database
    vector_db = build_or_load_db()

    # Test it out!
    # ask_assistant(vector_db, "I'm getting an 'Invalid account or partition' error when submitting an sbatch job.")
    
    # You can uncomment this to run an interactive loop:
    while True:
         query = input("\nAsk the HPC Assistant (or type 'exit'): ")
         if query.lower() == 'exit': break
         ask_assistant(vector_db, query)
