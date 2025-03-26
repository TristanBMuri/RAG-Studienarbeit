import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time

# Configuration
base_url = "https://www.concordia.ch"
output_dir = "company_pdfs"
max_depth = 5
delay = 1.5
headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyCrawler/1.0)'}

# Initialize
visited_urls = set()
pdf_urls = set()
queue = deque([(base_url, 0)])
domain = urlparse(base_url).netloc


def is_valid(url):
    parsed = urlparse(url)
    return parsed.netloc == domain


def is_pdf(url):
    return urlparse(url).path.lower().endswith('.pdf')


def download_pdf(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(url).path)
            filepath = os.path.join(output_dir, filename)

            # Overwrite existing file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded/Overwritten: {filename}")
        else:
            print(f"❌ Failed: {url} (HTTP {response.status_code})")
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {str(e)}")


def crawl_site():
    os.makedirs(output_dir, exist_ok=True)

    while queue:
        url, depth = queue.popleft()

        if depth > max_depth:
            continue

        try:
            print(f"🔍 Crawling: {url} (depth {depth})")
            time.sleep(delay)

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)
                    clean_url = parsed_url._replace(fragment="", query="").geturl()

                    if is_pdf(clean_url):
                        if clean_url not in pdf_urls:
                            pdf_urls.add(clean_url)
                            print(f"📄 Found PDF: {clean_url}")
                            download_pdf(clean_url)
                    elif is_valid(clean_url) and clean_url not in visited_urls:
                        visited_urls.add(clean_url)
                        queue.append((clean_url, depth + 1))
        except Exception as e:  # This was missing proper indentation
            print(f"⚠️ Error crawling {url}: {str(e)}")


if __name__ == "__main__":
    crawl_site()
    print("\nCrawling complete!")
    print(f"Total PDFs found: {len(pdf_urls)}")
    print(f"PDFs saved to: {os.path.abspath(output_dir)}")