import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io
import time
import re
import json
from PyPDF2 import PdfReader
import os

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from a PDF (given as bytes) using PyPDF2.
    """
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text):
    """
    Perform basic cleaning: remove excessive newlines and fix hyphenated line breaks.
    """
    cleaned = re.sub(r'\n+', '\n', text)
    cleaned = cleaned.replace('-\n', '')
    return cleaned.strip()

def segment_text(text):
    """
    Segment text into sections based on common headings.
    Returns a dictionary mapping section names to their text.
    """
    # List of common section headers
    sections = ["Abstract", "Introduction", "Case Presentation", "Background",
                "Discussion", "Conclusion", "Methods", "Results"]
    section_dict = {}
    for sec in sections:
        pattern = r'\b' + re.escape(sec) + r'\b'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            section_dict[sec] = matches[0].start()
    sorted_sections = sorted(section_dict.items(), key=lambda x: x[1])
    segmented = {}
    for i, (sec, start) in enumerate(sorted_sections):
        end = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
        segmented[sec] = text[start:end].strip()
    return segmented

def extract_metadata_from_pdf_bytes(pdf_bytes):
    """
    Extract metadata from the PDF using PyPDF2.
    """
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    metadata = reader.metadata
    return dict(metadata) if metadata else {}

def main():
    # Base URL and target page URL
    base_url = "https://jmedicalcasereports.biomedcentral.com"
    page_url = (
        "https://jmedicalcasereports.biomedcentral.com/articles"
        "?searchType=journalSearch&sort=PubDate&page=2"
    )
    
    # Initialize a session with a realistic User-Agent
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    })
    
    print(f"Fetching article listing from: {page_url}")
    resp = session.get(page_url)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Extract article links (BMC uses <a data-test="title-link">)
    article_links = []
    for a_tag in soup.find_all("a", {"data-test": "title-link"}, href=True):
        href = a_tag["href"]
        if href.startswith("/articles/"):
            article_links.append(urljoin(base_url, href))
    article_links = list(set(article_links))
    print(f"Found {len(article_links)} article links on page 2.")
    
    # Directory for saving JSON output files
    json_dir = "extracted_json"
    os.makedirs(json_dir, exist_ok=True)
    
    # Process each article
    for article_url in article_links:
        print(f"\nProcessing article: {article_url}")
        try:
            article_resp = session.get(article_url)
            article_resp.raise_for_status()
            article_soup = BeautifulSoup(article_resp.text, "html.parser")
            
            # Look for the PDF download link using possible selectors.
            pdf_tag = article_soup.find("a", {"data-test": "pdf-link"}, href=True)
            if not pdf_tag:
                pdf_tag = article_soup.find("a", {"data-track-action": "download pdf"}, href=True)
            
            if pdf_tag:
                raw_pdf_url = pdf_tag["href"]
                pdf_url = urljoin(article_url, raw_pdf_url)
                print(f"Found PDF link: {pdf_url}")
                
                pdf_resp = session.get(pdf_url)
                pdf_resp.raise_for_status()
                pdf_bytes = pdf_resp.content
                
                # Extract, clean, and segment text from the PDF.
                full_text = extract_text_from_pdf_bytes(pdf_bytes)
                cleaned_text = clean_text(full_text)
                sections = segment_text(cleaned_text)
                metadata = extract_metadata_from_pdf_bytes(pdf_bytes)
                
                # We do not store the full text; only the segmented sections (plus basic metadata)
                output_data = {
                    "article_url": article_url,
                    "pdf_url": pdf_url,
                    "metadata": metadata,
                    "sections": sections
                }
                
                # Create a JSON filename based on the PDF filename.
                base_filename = pdf_url.split("/")[-1]
                if base_filename.lower().endswith(".pdf"):
                    base_filename = base_filename[:-4]
                json_filename = base_filename + ".json"
                json_filepath = os.path.join(json_dir, json_filename)
                
                with open(json_filepath, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved extracted JSON to: {json_filepath}")
            else:
                print("No PDF link found on this article page.")
        except Exception as e:
            print(f"Error processing {article_url}: {e}")
        time.sleep(2)

if __name__ == "__main__":
    main()