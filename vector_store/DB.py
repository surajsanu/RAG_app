from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document


load_dotenv()
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5"
)

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]


vector_store = Chroma.from_documents(
    collection_name="langchain_db",
    documents=docs,
    embedding=hf_embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

result = vector_store.similarity_search("What is used for data analysis?", k=1)

for doc in result:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")