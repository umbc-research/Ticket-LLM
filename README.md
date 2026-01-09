UMBC HPC Support Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system built to help HPC Sysadmins troubleshoot cluster issues using institutional knowledge from 3,000+ past support tickets.
🚀 Quick Start (HPC Workflow)

Follow these steps to get the assistant running on a GPU compute node:

    Request a GPU Node:
    Bash

srun --cluster=chip-gpu --account=pi_doit --mem=50G --time=8:00:00 --gres=gpu:1 --pty $SHELL

Initialize Ollama:
Bash

module load ollama/0.13.5
ollama serve > ollama_server.log 2>&1 &

Activate Environment & Run:
Bash

    conda activate hpc_assistant
    python rag_search.py

📂 File Overview
File	Purpose
rag_search.py	The main application. Handles retrieval logic, database loading, and LLM interaction.
clean_tickets.py	Pre-processes raw ticketing CSVs—strips HTML, merges threads by ID, and prepares the "Context."
check_Database_Size.py	Utility script to verify the number of text chunks currently indexed (Target: ~5,400).
hpc_70b.Modelfile	The "recipe" for the llama-70b-hpc model, including the 32k context window configuration.
wiki_map.txt	A high-density reference file mapping keywords to official UMBC Wiki links.
cleaned_tickets.csv	The primary knowledge source generated from the raw ticketing data.
ticket_db/	The Chroma vector database folder containing all indexed ticket embeddings.
🛠️ Configuration & Customization
The 70B Model

This setup uses a 70-Billion parameter Llama 3.1 model.

    Memory Requirement: ~43GB of VRAM.

    Loading Time: Takes ~60-90 seconds to load into the GPU on initial query.

Updating Knowledge

To add new tickets or update the wiki_map.txt:

    Delete the current database: rm -rf ticket_db/

    Run rag_search.py.

    The script will automatically detect the missing folder and re-index the CSV/Wiki Map (approx. 15-minute process).

💡 Pro-Tips for the Team

    Interactive Mode: The assistant stays in a loop. Type exit to quit and free up the GPU.

    Troubleshooting Stalls: If the assistant stops responding, check ollama_server.log or run nvidia-smi to see if the process is actually active or if the GPU memory is full.
