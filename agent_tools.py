import httpx
from bs4 import BeautifulSoup  # HTML parser
from typing import List, Dict

def parse_html(html_content: str) -> List[Dict[str, Any]]:
    """Parse the HTML content from arXiv to extract relevant information."""
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = []
    for art in soup.find_all('div', class_='arxiv-result'):
        title = art.find('p', class_='title').text.strip()
        authors = [a.text.strip() for a in art.find('p', class_='authors').find_all('a')]
        arxiv_id = art['id'].split('/')[-1]
        abstract_url = art.find('p', class_='abstract').find('a')['href']
        pdf_url = art.find('p', class_='download').find('a')['href']
        subjects = [s.text for s in art.find_all('span', class_='subject')]
        primary_subject = subjects[0] if subjects else ''

        articles.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract_url": abstract_url,
            "pdf_url": pdf_url,
            "subjects": subjects,
            "primary_subject": primary_subject
        })
    return articles