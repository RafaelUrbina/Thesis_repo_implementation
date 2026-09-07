"""
Converts PDF files to plain text, automatically handling both digital (text-based)
and scanned (image-based) documents.

This script implements a robust pipeline that:
1.  Detects if a PDF page contains selectable text using PyMuPDF.
2.  If text is present, it extracts it directly for high speed and accuracy.
3.  If a page is image-based (scanned), it renders the page as an image.
4.  Performs Optical Character Recognition (OCR) on the image using Tesseract.
5.  Cleans the extracted text to remove common OCR artifacts.
6.  Saves the final, cleaned text to a .txt file.
"""

import sys
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Add the repository root to the Python path for module imports
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utils.paths import CAUSAL_LINKS_PATH

# --- Tesseract Configuration ---
# On Windows, you must specify the path to the Tesseract executable file.
# Update this path if you installed Tesseract in a different location.
# The 'r' before the string is important and handles backslashes correctly.
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\urbi1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


def clean_text(text: str) -> str:
    """
    Performs basic cleaning on extracted text.

    -   Removes words broken by hyphens at line breaks.
    -   Reduces multiple consecutive newlines to a maximum of two.
    -   Strips leading/trailing whitespace.
    """
    # Fix hyphenated words at the end of a line
    text = re.sub(r"(\w)-(\s+)\n(\w)", r"\1\3", text)
    # Reduce 3 or more newlines to just two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_page(page: fitz.Page) -> str:
    """Processes a single PDF page, extracting text via direct extraction or OCR."""
    # Try direct text extraction first
    text = page.get_text().strip()

    # If text is minimal (likely a scanned page), perform OCR
    if len(text) < 100:
        try:
            # Render page to an image at 300 DPI for better OCR quality
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Use Tesseract to OCR the image
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            return ocr_text
        except Exception as e:
            # Return a marker if OCR fails on this page
            return f"[OCR Error on page {page.number + 1}: {e}]"
    return text


def convert_pdf_to_txt(pdf_path: Path, output_path: Path):
    """
    Converts a single PDF to a text file, handling both digital and scanned content.

    Args:
        pdf_path (Path): The path to the source PDF file.
        output_path (Path): The path where the output TXT file will be saved.
    """
    if not pdf_path.is_file():
        print(f"Error: PDF not found at {pdf_path}")
        return

    print(f"--- Processing: '{pdf_path.name}' ---")
    full_text = []

    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            print(f"  - Processing page {i + 1}/{len(doc)}...")
            page_text = process_page(page)
            full_text.append(page_text)
        doc.close()

        # Join text from all pages and perform final cleaning
        final_text = clean_text("\n\n---\n\n".join(full_text))

        # Save the final output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Successfully saved text to '{output_path}'")

    except Exception as e:
        print(f"An error occurred while processing '{pdf_path.name}': {e}")


def process_file_wrapper(args):
    """Helper function to unpack arguments for ProcessPoolExecutor."""
    pdf_path, output_dir = args
    output_path = output_dir / f"{pdf_path.stem}.txt"
    convert_pdf_to_txt(pdf_path, output_path)


if __name__ == "__main__":
    # --- Configuration for batch processing ---
    pdf_input_dir = CAUSAL_LINKS_PATH / "causal_texts" / "pdf"
    text_output_dir = CAUSAL_LINKS_PATH / "causal_texts" / "txt"

    # --- Script execution ---
    # Ensure the output directory exists
    text_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for PDF files in: {pdf_input_dir}")
    all_pdf_files = list(pdf_input_dir.glob("*.pdf"))

    if not all_pdf_files:
        print("No PDF files found to process.")
    else:
        # Filter out PDFs that have already been converted
        pdf_files_to_process = [
            pdf for pdf in all_pdf_files
            if not (text_output_dir / f"{pdf.stem}.txt").exists()
        ]

        print(f"Found {len(all_pdf_files)} total PDFs. {len(pdf_files_to_process)} new PDFs to convert.")

        # Prepare arguments for parallel processing
        tasks = [(pdf_path, text_output_dir) for pdf_path in pdf_files_to_process]

        # Use ProcessPoolExecutor for parallel processing to speed up OCR
        # Set max_workers to None to use all available CPU cores
        with ProcessPoolExecutor(max_workers=None) as executor:
            # The map function will distribute the tasks to the worker processes
            executor.map(process_file_wrapper, tasks)

        print("\n--- All PDF processing complete. ---")