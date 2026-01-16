#!/bin/bash
#SBATCH --job-name=ticket_cleaner
#SBATCH --output=logs/clean_%j.out
#SBATCH --error=logs/clean_%j.err
#SBATCH --cluster=chip-gpu
#SBATCH --account=pi_doit
#SBATCH --partition=gpu     # Standard GPU partition

# --- GPU CONFIGURATION ---
# Note: You used --gres=gpu:1 in your test.
# IF you are using the 70B model, 1 GPU might run out of memory (OOM) 
# unless it's an 80GB A100.
# I have kept it at 2 GPUs for safety. If the scheduler rejects it, change to gpu:1.
#SBATCH --gres=gpu:2        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# --- 1. ENVIRONMENT SETUP ---
echo "Job started on $(hostname) at $(date)"

# Load the exact modules you used in your interactive session
echo "Loading modules..."
module purge
module load ollama/0.13.5
# You might need to load conda/anaconda if 'conda' isn't found, 
# but usually it's in the base path.

# Initialize Conda for the script (this is required for 'conda activate' to work in scripts)
# We assume standard conda setup. If this fails, try: source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate your specific environment
echo "Activating Conda environment: hpc_assistant"
conda activate hpc_assistant

# --- 2. START OLLAMA SERVER ---
echo "Starting local Ollama server..."
# Point to your models directory to ensure it finds the 70B model
export OLLAMA_MODELS=~/.ollama/models  

# Start the server in the background
ollama serve > logs/ollama_server_$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!

# Give it time to initialize (30s is safer for bigger models)
sleep 30
echo "Ollama server active (PID: $OLLAMA_PID)"

# --- 3. RUN THE CLEANER ---
echo "Running python script..."
# Ensure unbuffered output (-u) so you see print statements in the log file immediately
python -u cleanData.py

# --- 4. CLEANUP ---
echo "Job finished at $(date). Stopping Ollama..."
kill $OLLAMA_PID
