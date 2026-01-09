import pandas as pd
import re
import html
import csv

def final_polish_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Standard cleaning (HTML and Entities)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Aggressive Form Removal (The "First Name... Campus ID" block)
    form_pattern = r'(First Name|Last Name|EMail|Campus ID|Subject|Alternate Email):.*?(?=(Hi|Hello|Dear|I am|I have|The destination|$))'
    text = re.sub(form_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # 3. CRITICAL: Remove all actual newlines and carriage returns
    # This prevents the "Runoff" into the ID column
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 4. Remove Email Quoted Lines (fragments starting with >)
    # Since we removed newlines, we do this line-by-line BEFORE replacing them
    # But since we want a clean block, let's just use a regex for common patterns
    text = re.sub(r'On\s.*wrote:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'>+', ' ', text)

    # 5. Remove Staff Signatures
    text = re.sub(r'Roy Prouty\s+UMBC Office:.*?\d{3}-\d{3}-\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'--\s*.*?(Matthias Gobbert|Roy Prouty|Director|Professor).*', '', text, flags=re.IGNORECASE)

    # 6. Final cleanup: Remove double-quotes and extra spaces
    # Double quotes inside the text can confuse CSV readers
    text = text.replace('"', "'") 
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_cleaning(input_csv, output_csv):
    print(f"Reading {input_csv}...")
    # Read the file. If it has runoffs, we only want the first 15 columns 
    # and we drop any row where 'TicketID' isn't a number.
    df = pd.read_csv(input_csv, low_memory=False)

    # Clean the ID column: remove rows that were created by 'runoffs'
    # Valid TicketIDs are strictly numeric
    df['TicketID'] = pd.to_numeric(df['TicketID'], errors='coerce')
    df = df.dropna(subset=['TicketID'])
    df['TicketID'] = df['TicketID'].astype(int)

    print("Cleaning and Flattening Text...")
    df['TransactionContent'] = df['TransactionContent'].apply(final_polish_text)

    # Filter out empty/useless transactions
    df = df[df['TransactionContent'].str.len() > 10]

    print("Grouping into threads...")
    df_final = df.groupby('TicketID').agg({
        'CreatedDate': 'first',
        'SubjectNoHTML': 'first',
        'Requestor': 'first',
        'TransactionContent': lambda x: " [THREAD]: ".join(dict.fromkeys(x))
    }).reset_index()

    print(f"Saving to {output_csv}...")
    # Using QUOTE_MINIMAL is safe now because we've removed all internal newlines
    df_final.to_csv(output_csv, 
                    index=False, 
                    quoting=csv.QUOTE_ALL, 
                    escapechar='\\', 
                    encoding='utf-8')
    
    print("Success! Runoffs eliminated.")

if __name__ == "__main__":
    run_cleaning('allDoitTickets.csv', 'cleaned_tickets.csv')
