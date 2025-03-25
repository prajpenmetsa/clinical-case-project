import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time

def main():
    # Base URL for building absolute links (if needed)
    base_url = "https://jmedicalcasereports.biomedcentral.com"
    
    # The specific page (page=2) you want to scrape
    page_url = (
        "https://jmedicalcasereports.biomedcentral.com/articles"
        "?searchType=journalSearch&sort=PubDate&page=2"
    )
    
    # Use a requests Session for consistent headers/cookies
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
    
    # Parse the listing page
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find all article links (BMC typically uses <a data-test="title-link">)
    article_links = []
    for a_tag in soup.find_all("a", {"data-test": "title-link"}, href=True):
        href = a_tag["href"]
        # Should look like: /articles/10.1186/s13256-024-05000-5
        if href.startswith("/articles/"):
            article_links.append(urljoin(base_url, href))
    
    # Deduplicate
    article_links = list(set(article_links))
    print(f"Found {len(article_links)} article links on page 2.")
    
    # Prepare a folder to store downloaded PDFs
    output_dir = "downloaded_articles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visit each article page
    for article_url in article_links:
        print(f"\nProcessing article: {article_url}")
        try:
            article_resp = session.get(article_url)
            article_resp.raise_for_status()
            article_soup = BeautifulSoup(article_resp.text, "html.parser")
            
            # Look for the PDF link. On BMC, it's often:
            #   <a data-test="pdf-link" ...> or <a data-track-action="download pdf" ...>
            pdf_tag = article_soup.find("a", {"data-test": "pdf-link"}, href=True)
            if not pdf_tag:
                # Fallback if data-test="pdf-link" doesn't exist
                pdf_tag = article_soup.find("a", {"data-track-action": "download pdf"}, href=True)
            
            if pdf_tag:
                raw_pdf_url = pdf_tag["href"]
                
                # Resolve the final absolute PDF URL (avoids double domain)
                pdf_url = urljoin(article_url, raw_pdf_url)
                
                print(f"Found PDF link: {pdf_url}")
                
                # Download the PDF
                pdf_resp = session.get(pdf_url)
                pdf_resp.raise_for_status()
                
                # Derive a filename from the URL
                filename = pdf_url.split("/")[-1]
                if not filename.lower().endswith(".pdf"):
                    filename += ".pdf"
                
                # Save it locally
                file_path = os.path.join(output_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(pdf_resp.content)
                
                print(f"Downloaded PDF to: {file_path}")
            else:
                print("No PDF link found on this article page.")
        
        except Exception as e:
            print(f"Error processing {article_url}: {e}")
        
        # Be polite and not too fast
        time.sleep(2)

if __name__ == "__main__":
    main()