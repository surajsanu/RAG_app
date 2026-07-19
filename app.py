import os
import tempfile

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from dotenv import load_dotenv


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )


embedding_model = get_embedding_model()


# --------------------------------------------------
# LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile"
    )


llm = get_llm()


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's question using ONLY the provided context.

        If the answer is not present in the context, say:
        "I could not find the answer in the document."

        Give clear and well-structured answers.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


# --------------------------------------------------
# Extract PDF
# --------------------------------------------------

def load_pdf(uploaded_file):

    documents = []

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        reader = PdfReader(temp_path)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            # Skip empty pages
            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num + 1
                        }
                    )
                )

    finally:

        # Delete temporary PDF
        os.remove(temp_path)

    return documents


# --------------------------------------------------
# Split Documents
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

def create_vector_store(chunks):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📚 PDF RAG")

    st.write(
        "Upload a PDF book and ask questions "
        "based on its content."
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        # Process only if a new PDF is uploaded
        if st.session_state.current_file != uploaded_file.name:

            if st.button(
                "Process PDF",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Reading and processing your PDF..."
                ):

                    try:

                        # Load PDF
                        documents = load_pdf(uploaded_file)

                        if not documents:
                            st.error(
                                "No readable text was found in the PDF."
                            )

                        else:

                            # Split documents
                            chunks = split_documents(documents)

                            # Create embeddings + vector store
                            vector_store = create_vector_store(
                                chunks
                            )

                            # Save in session
                            st.session_state.vector_store = (
                                vector_store
                            )

                            st.session_state.current_file = (
                                uploaded_file.name
                            )

                            # Clear old chat
                            st.session_state.messages = []

                            st.success(
                                f"PDF processed successfully!"
                            )

                            st.info(
                                f"""
                                Pages: {len(documents)}

                                Chunks: {len(chunks)}
                                """
                            )

                    except Exception as e:

                        st.error(
                            f"Error processing PDF: {e}"
                        )

    if st.session_state.vector_store:

        st.divider()

        if st.button(
            "Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("💬 Chat With Your PDF")

st.caption(
    "Upload a PDF book and ask questions about its content."
)


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

if st.session_state.vector_store is None:

    st.info(
        "👈 Upload and process a PDF from the sidebar to start chatting."
    )

else:

    query = st.chat_input(
        "Ask something about your PDF..."
    )

    if query:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Display user message
        with st.chat_message("user"):

            st.markdown(query)


        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document..."
            ):

                # Create retriever
                retriever = (
                    st.session_state
                    .vector_store
                    .as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                )

                # Retrieve documents
                retrieved_docs = retriever.invoke(
                    query
                )


                # Create context
                context = "\n\n".join(

                    [
                        f"""
                        Page: {doc.metadata.get('page')}

                        {doc.page_content}
                        """

                        for doc in retrieved_docs
                    ]
                )


                # Create final prompt
                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )


                # Call LLM
                response = llm.invoke(
                    final_prompt
                )


                answer = response.content


                # Display answer
                st.markdown(answer)


                # Show retrieved sources
                with st.expander(
                    "View Sources"
                ):

                    for i, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        st.markdown(
                            f"""
                            **Source {i} — Page {
                                doc.metadata.get(
                                    'page',
                                    'Unknown'
                                )
                            }**

                            {doc.page_content[:500]}...
                            """
                        )

                        st.divider()


        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )