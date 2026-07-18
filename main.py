from dotenv import load_dotenv
# from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
client = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    
)

## Document Loader for text files
# text = Path("document_loaders/notes.txt").read_text()

# docs = [Document(page_content=text)]

## Document Loader for pdf files
reader = PdfReader("document_loaders/deeplearning.pdf")


docs = []

for page_num, page in enumerate(reader.pages):
    text = page.extract_text() or ""

    docs.append(
        Document(
            page_content=text,
            metadata={
                "source": "deeplearning.pdf",
                "page": page_num
            }
        )
    )


## Implementing RecursiveCharacterTextSplitter for splitting the document into chunks of 1000 characters with an overlap of 200 characters. The separators used for splitting are double newlines, single newlines, spaces, and empty strings. This allows for more natural breaks in the text, preserving context and meaning across chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
template= ChatPromptTemplate.from_messages([
    ("system","You are a high precision data summarizer that summarizes a document in 50 words."),
    ("human","{data}")
    ])
model = ChatHuggingFace(llm=client)

