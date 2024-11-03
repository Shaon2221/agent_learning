import os
import streamlit as st
from typing import List, Dict, Any
import tempfile

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class RAGSystem:
    def __init__(
        self,
        model_name: str = "llama3.1",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Ollama with ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            temperature=0
        )
        
        # Create prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.

        Context: {context}

        Question: {question}

        Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def process_file(self, uploaded_file) -> List[Dict]:
        """Process a single PDF file"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            splits = text_splitter.split_documents(pages)

            # Add source filename to metadata
            for split in splits:
                split.metadata["source"] = uploaded_file.name

            return splits

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    def setup_rag_chain(self, vector_store: Chroma) -> RetrievalQA:
        """Create the RAG chain"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )


def initialize_session_state():
    """Initialize session state variables"""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag" not in st.session_state:
        st.session_state.rag = RAGSystem()


def main():
    st.set_page_config(
        page_title="PDF Chat",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Chat with your PDFs")
    
    # Initialize session state
    initialize_session_state()

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            all_docs = []
            new_files = False

            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            docs = st.session_state.rag.process_file(uploaded_file)
                            all_docs.extend(docs)
                            st.session_state.processed_files.add(uploaded_file.name)
                            new_files = True
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            if new_files and all_docs:
                with st.spinner("Updating knowledge base..."):
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=all_docs,
                        embedding=st.session_state.rag.embeddings,
                    )
                st.success("PDFs processed successfully!")

        # Show processed files
        if st.session_state.processed_files:
            st.write("Processed files:")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

    # Main chat interface
    if not st.session_state.vector_store:
        st.info("Please upload PDF files to start chatting.")
    else:
        # Create QA chain
        qa_chain = st.session_state.rag.setup_rag_chain(st.session_state.vector_store)

        # Question input
        question = st.text_input("Ask a question about your documents:")

        if question:
            # Create a container for streaming output
            answer_container = st.empty()
            stream_handler = StreamHandler(answer_container)
            
            # Add streaming callback to the LLM
            st.session_state.rag.llm.callbacks = [stream_handler]

            try:
                # Get response
                response = qa_chain.invoke({"query": question})
                
                # Display sources
                st.subheader("Sources:")
                for doc in response["source_documents"]:
                    with st.expander(
                        f"Source: {doc.metadata['source']} "
                        f"(Page {doc.metadata['page'] + 1})"
                    ):
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 