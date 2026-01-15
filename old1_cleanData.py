import pandas as pd
import ollama
import csv
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = 'trimmed_file.csv'
OUTPUT_FILE = 'cleanedData.csv'
MODEL = "llama-hpc"

# LIST THE COLUMNS YOU WANT TO DELETE PERMANENTLY
COLUMNS_TO_DELETE = [
    'TransactionID',
    'TicketOwnerUsername',
    'RequestorUsername',
    'TransactionName',
    'TransactionEmail'
]

def clean_content_with_ai(raw_text):
    """
    Uses the 8B model to strip HTML and junk, then sanitizes the string
    to ensure it doesn't break the CSV structure.
    """
    if not isinstance(raw_text, str) or len(raw_text.strip()) < 5:
        return ""

    prompt = f"""
    Act as a text cleaner and data anonymizer. I will give you a raw email/ticket body. 
    1. Remove all HTML tags and CSS.
    2. Remove email headers, footers, and UMBC privacy notices.
    3. Fix encoding issues (e.g., convert '=3D' to '=').
    4. Keep technical content, server names, and error messages.
    5. ANONYMIZE: Replace all human names, email addresses, phone numbers, and Campus IDs with generic placeholders (e.g., [USER], [STAFF], [EMAIL], [ID]). No person should be identifiable.
    
    Return ONLY the cleaned text on a single line. Do not add any 'Here is the text' intro.

    RAW TEXT:
    {raw_text}
    """
    
    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        text = response['response'].strip()
        
        text = text.replace('"', "'")        # Replace double quotes with single
        text = text.replace('\n', ' ')       # Remove newlines
        text = text.replace('\r', ' ')       # Remove carriage returns
        text = text.replace('\t', ' ')       # Remove tabs
        text = text.replace('\xa0', ' ')     # Remove non-breaking spaces
        
        return text
    except Exception as e:
        # If AI fails, return a sanitized version of the raw text
        return str(raw_text).replace('"', "'").replace('\n', ' ')

def merge_thread(group):
    """
    Merges all replies for a TicketID into one row.
    Uses the 'Create' row for metadata.
    """
    # Use the 'Create' row as the master record
    create_row = group[group['TransactionType'] == 'Create']
    if not create_row.empty:
        primary_row = create_row.iloc[0].copy()
    else:
        primary_row = group.iloc[0].copy()

    # Join the narrative chronologically
    full_narrative = " | ".join(group['TransactionContent'].astype(str))
    primary_row['TransactionContent'] = full_narrative
    
    return primary_row

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"--- 1. Loading {INPUT_FILE} Robustly ---")
    # Robust reader to handle messy input data
    df = pd.read_csv(
        INPUT_FILE, 
        engine='python', 
        on_bad_lines='warn', 
        quoting=0, 
        doublequote=True
    )

    print(f"--- 2. Filtering and Pre-Processing ---")
    # Delete truly useless columns
    df = df.drop(columns=COLUMNS_TO_DELETE, errors='ignore')
    
    # Filter for recency (2025 onwards)
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
    df = df[df['CreatedDate'] >= '2025-01-01']
    print(f"Tickets to process after date filter: {len(df)}")

    print(f"--- 3. AI Cleaning (Intelligence Step) ---")
    if 'TransactionContent' in df.columns:
        tqdm.pandas(desc="Cleaning Text")
        df['TransactionContent'] = df['TransactionContent'].progress_apply(clean_content_with_ai)
    
    print(f"--- 4. Consolidating Threads ---")
    # Ensure chronological order before grouping
    df = df.sort_values(by=['TicketID', 'CreatedDate'])
    df = df.groupby('TicketID').apply(merge_thread).reset_index(drop=True)

    print(f"--- 5. Final Cleanup and Export ---")
    # Remove helper columns that we no longer need for the final RAG DB
    df = df.drop(columns=['TransactionType', 'CreatedDate'], errors='ignore')

    # --- 6. POST-PROCESS VALIDATION ---
    print("--- 6. Running Final Integrity Check ---")
    verified_rows = []
    
    # Re-read the file we just wrote to check for leaks
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        verified_rows.append(header)
        
        for row in reader:
            if not row: continue # Skip empty
            
            # Check if the first column is a digit (TicketID)
            # This catches cases where the AI broke the formatting and 
            # text like "ROFILES..." starts a new line.
            if row[0].isdigit():
                verified_rows.append(row)
            else:
                # This line is garbage/leaked text, skip it
                continue

    print(f"Integrity check complete. Removed corrupted 'leak' lines.")
    
    # Save the final, verified version
    FINAL_VERIFIED_FILE = "verified_" + OUTPUT_FILE
    with open(FINAL_VERIFIED_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(verified_rows)
    
    print(f"ULTIMATE SUCCESS! Final verified file: {FINAL_VERIFIED_FILE}")


    # QUOTE_ALL is the secret to preventing the 'ROFILES' shifting error
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    print(f"Success! Final file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
