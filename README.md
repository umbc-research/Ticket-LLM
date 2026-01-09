## UMBC HPC Support Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system built to help HPC Sysadmins troubleshoot cluster issues using institutional knowledge from 3,000+ past support tickets.

### Quick Start (HPC Workflow)

Follow these steps to get the assistant running on a GPU compute node:

Request a GPU Node:
```
srun --cluster=chip-gpu --account=pi_doit --mem=50G --time=8:00:00 --gres=gpu:1 --pty $SHELL
```
Initialize Ollama:

```
module load ollama/0.13.5
ollama serve > ollama_server.log 2>&1 &
```
Activate Environment & Run:

```
conda activate hpc_assistant
python rag_search.py
```

### File Overview

| File / Directory | Description |
| :--- | :--- |
| `rag_search.py` | The main application. Handles retrieval logic, database loading, and LLM interaction. |
| `clean_tickets.py` | Pre-processes raw ticketing CSVs—strips HTML and merges threads by ID. |
| `check_Database_Size.py` | Utility script to verify the number of text chunks currently indexed. |
| `hpc_70b.Modelfile` | The configuration for the llama-70b-hpc model (32k context window). |
| `wiki_map.txt` | A high-density reference file mapping keywords to official UMBC Wiki links. |
| `cleaned_tickets.csv` | The primary knowledge source generated from the raw ticketing data. |
| `ticket_db/` | The Chroma vector database folder containing all indexed ticket embeddings. |


### Configuration & Customization
The 70B Model

This setup uses a 70-Billion parameter Llama 3.1 model.

Memory Requirement: ~43GB of VRAM.

Loading Time: Takes ~60-90 seconds to load into the GPU on initial query.

Updating Knowledge

To add new tickets or update the wiki_map.txt:

Delete the current database: 
```
rm -rf ticket_db/
```
Run
```
python rag_search.py.
```

The script will automatically detect the missing folder and re-index the CSV/Wiki Map (approx. 15-minute process).

### Github Quick-Guide:
```
git add .

git commit -m "Update"

git push
```
