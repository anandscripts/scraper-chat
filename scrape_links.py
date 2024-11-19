from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.schema import Document

# Scrape Links
async def scrape_links(url, visited=None):
    if visited is None:
        visited = set()

    links_to_visit = set()
    base_netloc = urlparse(url).netloc  

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            absolute_url = urljoin(url, a_tag['href'])
            normalized_url, _ = urldefrag(absolute_url)
            normalized_url = normalized_url.rstrip('/')

            # Only add links with the same base domain that haven't been visited
            if urlparse(normalized_url).netloc == base_netloc and normalized_url not in visited:
                links_to_visit.add(normalized_url)

        visited.add(url.rstrip('/'))

        for link in links_to_visit.copy():
            yield link
            # Change 'await scrape_links(link, visited)' to async for to iterate over the generator
            async for new_link in scrape_links(link, visited):
                yield new_link

        # Yield the links instead of returning them
        for link in links_to_visit:
            yield link

    except requests.exceptions.RequestException as e:
        yield f"Error occurred while fetching the webpage: {e}\n"

# Scrape Text as Documents
def scrape_text(data):
    if type(data) == list:
        loader = SeleniumURLLoader(data)
        docs = loader.load()
    elif type(data) == str:
        docs = [Document(page_content=data)]
    return docs
