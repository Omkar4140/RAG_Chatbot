import os
import tempfile
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from io import StringIO

# Document processing
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings and Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM and Retrieval
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import warnings
warnings.filterwarnings("ignore")

class RAGApplication:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.setup_components()
    
    def setup_components(self):
        """Initialize embeddings and LLM"""
        try:
            # Initialize embeddings (free and efficient)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize Groq LLM
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("Please set your GROQ_API_KEY in the environment variables")
                return
                
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="mixtral-8x7b-32768",  # Fast and capable model
                temperature=0.1,
                max_tokens=1024
            )
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load based on file type
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    loader = PyPDFLoader(tmp_file_path)
                elif file_extension == 'txt':
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                elif file_extension == 'csv':
                    loader = CSVLoader(tmp_file_path)
                elif file_extension in ['doc', 'docx']:
                    loader = UnstructuredWordDocumentLoader(tmp_file_path)
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        try:
            if not documents:
                st.error("No documents to process")
                return
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create contextual compression retriever for better results
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever
            )
            
            st.success(f"Vector store created with {len(documents)} document chunks")
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        if not self.compression_retriever:
            st.error("Please upload and process documents first")
            return
        
        # Custom prompt template
        prompt_template = """
        You are an intelligent assistant for an internal company knowledge base. 
        Use the following context to answer the question. If you don't find the answer in the context, 
        say "I don't have enough information to answer this question based on the provided documents."
        
        Be concise, accurate, and helpful. If the question is unclear, ask for clarification.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.compression_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a question"""
        if not self.qa_chain:
            return {"error": "Please upload and process documents first"}
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}"}
    
    def process_documents(self, uploaded_files):
        """Complete document processing pipeline"""
        with st.spinner("Loading documents..."):
            documents = self.load_documents(uploaded_files)
        
        if not documents:
            st.error("No documents were successfully loaded")
            return False
        
        with st.spinner("Chunking documents..."):
            chunks = self.chunk_documents(documents)
        
        with st.spinner("Creating vector store..."):
            self.create_vectorstore(chunks)
        
        with st.spinner("Setting up QA chain..."):
            self.setup_qa_chain()
        
        return True

# Sample document content for testing
SAMPLE_DOCUMENTS = {
    "Company Policy": """
    Company Leave Policy
    
    Annual Leave: All employees are entitled to 25 days of annual leave per year.
    Sick Leave: Employees can take up to 10 days of sick leave per year with medical certificate.
    Maternity Leave: 16 weeks of paid maternity leave is provided.
    Remote Work: Employees can work remotely up to 3 days per week with manager approval.
    """,
    
    "IT Guidelines": """
    IT Security Guidelines
    
    Password Policy: Passwords must be at least 8 characters long and include uppercase, lowercase, numbers, and special characters.
    VPN Access: All remote connections must use the company VPN.
    Software Installation: Only approved software can be installed on company devices.
    Data Backup: All important data must be backed up daily to the cloud storage.
    """,
    
    "HR FAQ": """
    Frequently Asked Questions - HR
    
    Q: How do I request time off?
    A: Submit a request through the employee portal at least 2 weeks in advance.
    
    Q: What is the dress code?
    A: Business casual is required for office days, casual attire is acceptable for remote work.
    
    Q: How do I report an issue with my manager?
    A: Contact HR directly at hr@company.com or use the anonymous reporting system.
    """
}