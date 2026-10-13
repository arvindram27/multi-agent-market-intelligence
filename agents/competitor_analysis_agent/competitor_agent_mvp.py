# Competitor Analysis Agent MVP
import requests
from bs4 import BeautifulSoup
import json

def fetch_competitor_info(url):
    """Fetches and extracts basic information from a competitor's webpage."""
    print(f"Fetching competitor info from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    data = {
        "url": url,
        "page_title": "",
        "main_heading_h1": "",
        "meta_description": "",
        "extracted_paragraphs_sample": [] # Store first few paragraphs
    }

    # Extract Page Title
    if soup.title and soup.title.string:
        data["page_title"] = soup.title.string.strip()

    # Extract Main Heading (H1)
    h1_tag = soup.find('h1')
    if h1_tag and h1_tag.string:
        data["main_heading_h1"] = h1_tag.string.strip()
    elif h1_tag:
         # If h1 has nested tags, try to get all text
        data["main_heading_h1"] = h1_tag.get_text(separator=' ', strip=True)


    # Extract Meta Description
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
    if meta_desc_tag and meta_desc_tag.get('content'):
        data["meta_description"] = meta_desc_tag.get('content').strip()
    
    # Extract a sample of paragraph text (first 3 non-empty paragraphs)
    paragraphs = soup.find_all('p')
    count = 0
    for p in paragraphs:
        if p.get_text(strip=True):
            data["extracted_paragraphs_sample"].append(p.get_text(strip=True))
            count += 1
            if count >= 3:
                break
                
    print(f"- Extracted basic info for: {url}")
    return data

if __name__ == "__main__":
    print("--- Competitor Analysis Agent MVP Running ---")
    # Example usage: Using a well-known company's about page for demonstration
    # In a real application, this URL would be configurable and likely part of a list of competitors.
    # Using a placeholder that is generally accessible
    competitor_url = "https://www.google.com/about/"
    
    info = fetch_competitor_info(competitor_url)
    
    if info:
        print("\n--- Extracted Competitor Information (MVP) --- ")
        # Pretty print the JSON output
        print(json.dumps(info, indent=4))
    else:
        print("No information was extracted for the competitor.")
    print("\n--- Competitor Analysis Agent MVP Finished ---")

