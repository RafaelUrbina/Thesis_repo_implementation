import wikipediaapi
from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere in the project
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Always supply a descriptive User-Agent per Wikimedia API policy
wiki = wikipediaapi.Wikipedia(
    user_agent="FailureTermExtractor/1.0 (contact@example.com)",
    language="en"
)

def get_category_members(category_name, max_depth=1, current_depth=0, visited=None):
    if visited is None:
        visited = set()

    cat_page = wiki.page(f"Category:{category_name}")
    if not cat_page.exists():
        print(f"Category '{category_name}' not found.")
        return set()

    terms = set()
    
    # Iterate through members of the category
    for member in cat_page.categorymembers.values():
        # Exclude meta/discussion pages
        if member.ns in (0, 14):  # 0: Main articles, 14: Subcategories
            clean_title = member.title.replace("Category:", "").strip()
            terms.add(clean_title)

            # Traverse subcategories up to max_depth
            if member.ns == 14 and current_depth < max_depth and clean_title not in visited:
                visited.add(clean_title)
                terms.update(get_category_members(clean_title, max_depth, current_depth + 1, visited))

    return terms

# Categories to extract from
categories_to_extract = [
    #"Mechanical failure modes",
    #"Pipeline transport",
        #"Natural disasters",
        "Weather hazards",
        "Geological hazards"
]

all_terms = set()
for cat in categories_to_extract:
    print(f"Extracting terms from Category:{cat}...")
    extracted = get_category_members(cat, max_depth=1)
    all_terms.update(extracted)

# Import the path from the centralized paths.py
from Utils.paths import PATENTS_OUTPUT_TERMS_PATH
# Define the output directory and ensure it exists
PATENTS_OUTPUT_TERMS_PATH.mkdir(parents=True, exist_ok=True)
output_file = PATENTS_OUTPUT_TERMS_PATH / "[variable_terms]_wiki_variable_terms.txt"

# Save results to a clean text file
with open(output_file, "w", encoding="utf-8") as f:
    for term in sorted(all_terms):
        f.write(term + "\n")

print(f"\nDone! Extracted {len(all_terms)} terms to '{output_file}'.")