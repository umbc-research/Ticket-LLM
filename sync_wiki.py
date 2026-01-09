import requests
from bs4 import BeautifulSoup
import os
import time

# --- CONFIGURATION ---
PARENT_URL = "https://umbc.atlassian.net/wiki/spaces/faq/pages/1082589207/UMBC+HPCF+-+chip"
BASE_DOMAIN = "https://umbc.atlassian.net/wiki/spaces/faq/pages/1082589207/UMBC+HPCF+-+chip"
WIKI_CONTENT_DIR = "wiki_content"

def get_wiki_content(url):
    """Fetches a page and returns the title and the clean text content."""
    try:
        resp = requests.get(url)
        if resp.status_code != 200: return None, None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.find('title').text.split('-')[0].strip()
        
        # Confluence main content is usually in this div
        main_content = soup.find('div', {'id': 'main-content'})
        if not main_content: main_content = soup.find('body')
        
        return title, main_content.get_text(separator='\n')
    except:
        return None, None

def discover_child_urls(parent_url):
    """Finds all links on the page that belong to the same Confluence space."""
    links = set()
    try:
        resp = requests.get(parent_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Look for links that contain '/wiki/spaces/faq/pages/'
        # Note: Change 'faq' if your space name is different
        for a in soup.find_all('a', href=True):
            href = a['href']
            if "/wiki/spaces/faq/pages/" in href:
                # Build full URL if it's a relative path
                full_url = href if href.startswith('http') else BASE_DOMAIN + href
                # Strip fragments like #footer
                full_url = full_url.split('#')[0]
                links.add(full_url)
    except Exception as e:
        print(f"Error discovering links: {e}")
    return links

def sync():
    if not os.path.exists(WIKI_CONTENT_DIR): os.makedirs(WIKI_CONTENT_DIR)
    
    print(f"--- Starting Discovery from Parent: {PARENT_URL} ---")
    child_urls = discover_child_urls(PARENT_URL)
    # Add the parent itself to the list
    child_urls.add(PARENT_URL)
    
    print(f"Found {len(child_urls)} pages to sync. Starting download...")
    
    for url in child_urls:
        title, text = get_wiki_content(url)
        if title and text:
            # Clean filename
            safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).rstrip()
            with open(f"{WIKI_CONTENT_DIR}/{safe_title}.txt", "w") as f:
                f.write(f"SOURCE URL: {url}\nTITLE: {title}\n\n{text}")
            print(f"Saved: {title}")
            time.sleep(1) # Be nice to the server

if __name__ == "__main__":
    sync()
