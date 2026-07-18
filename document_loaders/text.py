from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path

text = Path("notes.txt").read_text()
# docs = [Document(page_content=text)]



# Implementing CharacterTextSplitter
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=10,
    chunk_overlap=0
)

docs = splitter.create_documents([text], metadatas=[{"source": "notes.txt"}])



print(docs[0].page_content)