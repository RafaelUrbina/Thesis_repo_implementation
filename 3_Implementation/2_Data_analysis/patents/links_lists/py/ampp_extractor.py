import sys
from pathlib import Path
import time
import requests
from bs4 import BeautifulSoup

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere in the project
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import PATENTS_OUTPUT_TERMS_PATH
BASE_URL = "https://www.ampp.org/technical-research/what-is-corrosion/corrosion-terminology-glossary/corrosion-terminology-"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Build the list: 'a' through 'w', plus the grouped 'xyz' suffix
url_suffixes = [chr(i) for i in range(ord('a'), ord('w') + 1)] + ["xyz"]

extracted_terms = set()

print("Starting extraction from AMPP Corrosion Glossary...")

for suffix in url_suffixes:
    url = f"{BASE_URL}{suffix}"
    print(f"Fetching: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Skipping suffix '{suffix}': HTTP status {response.status_code}")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Pull text from elements containing glossary definitions
        text_elements = soup.find_all(['li', 'p'])
        
        for element in text_elements:
            text = element.get_text(strip=True)
            
            # Match split delimiters used in AMPP entries
            if "—" in text:
                parts = text.split("—", 1)
                term = parts[0].strip()
                extracted_terms.add(term)
            elif " – " in text:
                parts = text.split(" – ", 1)
                term = parts[0].strip()
                extracted_terms.add(term)
                
    except Exception as e:
        print(f"Error scraping suffix {suffix}: {e}")
        
    time.sleep(1)

# Define the output directory and ensure it exists
PATENTS_OUTPUT_TERMS_PATH.mkdir(parents=True, exist_ok=True)
output_file = PATENTS_OUTPUT_TERMS_PATH / "[failure_terms]_ampp_failure.txt"

# Save results to a clean text file
with open(output_file, "w", encoding="utf-8") as f:
    for term in sorted(list(extracted_terms)):
        f.write(term + "\n")

print(f"\nDone! Extracted {len(extracted_terms)} terms to '{output_file}'.")