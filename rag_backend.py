import os
import tempfile
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# Multiple LLM providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as OpenRouterChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

class TFIDFEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        
        tfidf_matrix = self.vectorizer.transform(texts)
        return tfidf_matrix.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            return [0.0] * 1000
        
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()

class RAGApplication:
    def __init__(self):
        self.embeddings = TFIDFEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.api_provider = None
    
    def setup_llm(self, provider: str, api_key: str, model: str) -> bool:
        """Setup LLM based on provider"""
        try:
            if provider == "Groq":
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model,
                    temperature=0.1,
                    max_tokens=1024
                )
            elif provider == "OpenAI":
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name=model,
                    temperature=0.1,
                    max_tokens=1024
                )
            elif provider == "OpenRouter":
                self.llm = OpenRouterChat(
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name=model,
                    temperature=0.1,
                    max_tokens=1024
                )
            
            self.api_provider = f"{provider} ({model})"
            return True
            
        except Exception as e:
            st.error(f"Error setting up {provider}: {str(e)}")
            return False
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
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
                    continue
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
                
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vector store"""
        try:
            if not documents:
                return False
            
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Setup QA chain"""
        if not self.retriever or not self.llm:
            return False
        
        prompt_template = """
        You are an intelligent assistant. Use the following context to answer the question.
        If you don't find the answer in the context, say "I don't have enough information 
        to answer this question based on the provided documents."
        
        Context: {context}
        Question: {question}
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return True
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get answer for a question"""
        if not self.qa_chain:
            return {"error": "Please configure API and process documents first"}
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}"}
    
    def process_documents(self, uploaded_files) -> bool:
        """Process documents pipeline"""
        if not uploaded_files:
            return False
            
        documents = self.load_documents(uploaded_files)
        if not documents:
            return False
        
        chunks = self.chunk_documents(documents)
        
        if not self.create_vectorstore(chunks):
            return False
        
        return self.setup_qa_chain()
