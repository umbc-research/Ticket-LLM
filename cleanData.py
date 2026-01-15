import pandas as pd
import ollama
import csv
import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = 'trimmed_file.csv'   # Your main 3,000 ticket file
FINAL_FILE = 'cleanedData.csv'
TEMP_FILE = 'temp_draft_tickets.csv' # Temporary holding file
MODEL = "llama-hpc"

# LIST THE COLUMNS YOU WANT TO DELETE PERMANENTLY
# (TransactionType and CreatedDate are NOT here; we delete them at the very end)
COLUMNS_TO_DELETE = [
    'TransactionID',
    'TicketOwnerUsername',
    'RequestorUsername',
    'TransactionName',
    'TransactionEmail',
    'TicketOwner',
    'Requestor',
    'RequestorEmail',
    'TicketStatus',
    'QueueName'
]

def clean_content_with_ai(raw_text):
    """
    Uses the 8B model to strip HTML, Anonymize PII, and sanitize text.
    """
    if not isinstance(raw_text, str) or len(raw_text.strip()) < 5:
        return ""

    prompt = f"""
    Act as a text cleaner and data anonymizer. I will give you a raw email/ticket body. 
    1. Remove all HTML tags and CSS.
    2. Remove email headers, footers, and UMBC privacy notices.
    3. Fix encoding issues (e.g., convert '=3D' to '=').
    4. Keep technical content, server names, and error messages.
    5. ANONYMIZE: Replace all human names, email addresses, phone numbers, and Campus IDs with generic placeholders (e.g., [USER], [STAFF], [EMAIL]).
    
    Return ONLY the cleaned text on a single line. Do not add any 'Here is the cleaned text' intro.

    RAW TEXT:
    {raw_text}
    """

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        text = response['response'].strip()
        
        # SANITIZATION: Prevents CSV corruption
        text = text.replace('"', "'")        # No double quotes allowed
        text = text.replace('\n', ' ')       # No newlines
        text = text.replace('\r', ' ')       # No carriage returns
        text = text.replace('\t', ' ')       # No tabs
        
        return text
    except Exception as e:
        return ""

def merge_thread(group):
    """
    Merges all replies for a TicketID into one row.
    """
    create_row = group[group['TransactionType'] == 'Create']
    if not create_row.empty:
        primary_row = create_row.iloc[0].copy()
    else:
        primary_row = group.iloc[0].copy()

    # Join the narrative
    full_narrative = " | ".join(group['TransactionContent'].astype(str))
    primary_row['TransactionContent'] = full_narrative
    return primary_row

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. ROBUST LOAD
    print(f"--- 1. Loading {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='warn', quoting=0, doublequote=True)

    # 2. FILTER
    print(f"--- 2. Filtering ---")
    df = df.drop(columns=COLUMNS_TO_DELETE, errors='ignore')
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
    df = df[df['CreatedDate'] >= '2025-01-01']
    print(f"Tickets remaining: {len(df)}")

    # 3. AI CLEANING
    print(f"--- 3. AI Cleaning & Anonymizing ---")
    if 'TransactionContent' in df.columns:
        tqdm.pandas(desc="Processing Tickets")
        df['TransactionContent'] = df['TransactionContent'].progress_apply(clean_content_with_ai)
    
    # 4. CONSOLIDATE
    print(f"--- 4. Consolidating Threads ---")
    df = df.sort_values(by=['TicketID', 'CreatedDate'])
    df = df.groupby('TicketID').apply(merge_thread).reset_index(drop=True)

    # 5. SAVE DRAFT (The 'First Save')
    print(f"--- 5. Saving Draft to {TEMP_FILE} ---")
    df = df.drop(columns=['TransactionType'], errors='ignore')
    # We MUST save this so the validator can read it
    df.to_csv(TEMP_FILE, index=False, quoting=csv.QUOTE_ALL)

    # --- 6. VALIDATION (The "Look-Ahead" Integrity Check) ---
    print(f"--- 6. Running Precision Integrity Check ---")
    verified_rows = []
    dropped_count = 0
    
    with open(TEMP_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            verified_rows.append(header)
        except StopIteration:
            pass # Handle empty file
        
        # We hold the "previous" row in memory and only save it
        # once we confirm the "current" row is a valid new ticket.
        row_prev = None

        for row_curr in reader:
            if not row_curr: continue
            
            # Clean the ID to check it
            curr_id = row_curr[0].replace('"', '').strip()
            
            # THE RULE: Must be a number AND exactly 7 digits
            is_valid_ticket_start = curr_id.isdigit() and len(curr_id) == 7

            if is_valid_ticket_start:
                # The current row is a valid new ticket.
                # This confirms the PREVIOUS row ended correctly. Save it.
                if row_prev:
                    verified_rows.append(row_prev)
                # Set the current row as the new "pending" row
                row_prev = row_curr
            else:
                # The current row is GARBAGE (Row X).
                # ACTION: Delete Row X (curr) AND Row X-1 (prev).
                if row_prev:
                    # We discard the previous row (the one that likely leaked)
                    dropped_count += 1 
                
                # We discard the current garbage row
                dropped_count += 1
                
                # Reset buffer so we don't accidentally delete the innocent row before the pair
                row_prev = None 

        # If the very last row in the file was waiting, save it now
        if row_prev:
            verified_rows.append(row_prev)

    # 7. FINAL SAVE
    with open(FINAL_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(verified_rows)
    
    # Cleanup: Remove the temporary draft file
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    print(f"DONE! Dropped {dropped_count} corrupted lines.")
    print(f"Clean, Anonymized, and Verified data saved to: {FINAL_FILE}")

if __name__ == "__main__":
    main()
