import requests
from bs4 import BeautifulSoup

def scrape_text(links):
    data = ""
    for link in links:
        try:
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text(separator=' ', strip=True)
            data += page_text + "\n"
        except requests.exceptions.RequestException as e:
            continue
    return data