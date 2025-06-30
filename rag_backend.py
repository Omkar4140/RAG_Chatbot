import os
import tempfile
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Document processing
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Embeddings and Vector Store - Using TF-IDF instead of HuggingFace
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# LLM and Retrieval
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

class TFIDFEmbeddings(Embeddings):
    """Custom TF-IDF based embeddings to avoid HuggingFace rate limiting"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not self.fitted:
            # Fit the vectorizer on the first batch of documents
            self.vectorizer.fit(texts)
            self.fitted = True
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert to dense arrays and return as list of lists
        return tfidf_matrix.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.fitted:
            # If not fitted, return zeros (this shouldn't happen in normal flow)
            return [0.0] * 1000
        
        # Transform the query
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()

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
            # Initialize TF-IDF embeddings (no external API calls)
            self.embeddings = TFIDFEmbeddings()
            
            # Initialize Groq LLM with Mixtral model only
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("Please set your GROQ_API_KEY in the environment variables")
                return
                
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="mistral-saba-24b",  # Only Mixtral model
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
                # Add source metadata
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
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
            
            # Create vector store with TF-IDF embeddings
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            st.success(f"Vector store created with {len(documents)} document chunks")
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        if not self.retriever:
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
            retriever=self.retriever,
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
        if not uploaded_files:
            st.error("No files uploaded")
            return False
            
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
