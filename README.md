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
Move the ollama folder to your common directory, so nothing gets full:
```
mv ~/.ollama /umbc/rs/pi_group/users/user1/.ollama
ln -s /umbc/rs/pi_group/users/user1/.ollama ~/.ollama
```
Download the models needed:
```
ollama pull nomic-embed-text
ollama pull llama3.1:70b
```

Build the custom model:
```
ollama create llama-70b-hpc -f hpc_70b.Modelfile
```
Activate Environment & Run:

```
conda env create -f environment.yml
conda activate hpc_assistant
python rag_search.py
```

### File Overview

| File / Folder | Description |
| :--- | :--- |
| `rag_search.py` | The main application. Handles retrieval logic, database loading, and LLM interaction. |
| `clean_tickets.py` | Pre-processes raw ticketing CSVs—strips HTML and merges threads by ID. |
| `environment.yml` | (New) The Conda environment export. Allows teammates to recreate your setup. |
| `wiki_map.txt` | A high-density reference file mapping keywords to official UMBC Wiki links. |
| `hpc_70b.Modelfile` | The configuration for the high-end llama-70b-hpc model (requires ~43GB VRAM). |
| `hpc_8b.Modelfile` | (New) A lighter version for smaller GPUs (requires ~5.5GB VRAM). |
| `check_Database_Size.py` | Utility script to verify the number of text chunks currently indexed. |
| `README.md` | Project documentation and Quick Start guide. |
| `allDoitTickets.csv` | Raw ticketing data containing sensitive user information. |
| `cleaned_tickets.csv` | The processed text used to build the database. |
| `ticket_db/` | The folder containing the indexed vector embeddings. |
| `ollama_server.log` | Debug log for the Ollama background process. |


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
