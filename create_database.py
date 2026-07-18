# load pdf
# splt into chunks 
# create the embeddings
# store into chroma db

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

reader = PdfReader("./document_loaders/Deeplearning.pdf")
docs=[]
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

print("Loading Done")

## Implementing RecursiveCharacterTextSplitter for splitting the document into chunks of 1000 characters with an overlap of 200 characters. The separators used for splitting are double newlines, single newlines, spaces, and empty strings. This allows for more natural breaks in the text, preserving context and meaning across chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks= splitter.split_documents(docs) 
print("chunking done")

## Create embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)


print("embeddings got created")

vector_store = Chroma.from_documents(
    collection_name="langchain_db",
    documents=chunks,
    embedding=hf_embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)

print("vector store created")

