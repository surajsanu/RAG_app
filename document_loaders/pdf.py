from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter


# We created our own loader for PDF files because the langchain community is now deprecated from where we get PyPDFLoader. The default loader was not able to extract text from the PDF file. So we created our own loader that uses the pypdf library to extract text from the PDF file.
reader = PdfReader("GRU.pdf")


docs = []
for page_num, page in enumerate(reader.pages):
    text = page.extract_text() or ""

    docs.append(
        Document(
            page_content=text,
            metadata={
                "source": "GRU.pdf",
                "page": page_num
            }
        )
    )


# Implementing TokenTextSplitter
splitter = TokenTextSplitter(
    encoding_name="gpt2",
    chunk_size=1000,
    chunk_overlap=0
)

chunks = splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])



print(len(chunks))