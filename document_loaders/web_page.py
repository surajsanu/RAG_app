import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

url= "https://www.apple.com/in/macbook-pro/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

docs = [
    Document(
        page_content=soup.get_text(),
        metadata={"source": url}
    )
]

print(docs[0].page_content)