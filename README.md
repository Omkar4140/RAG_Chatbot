🤖 RAG Chatbot – AI-Powered Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) chatbot for answering questions from uploaded documents using advanced LLMs.

📌 Overview
This Streamlit-based RAG chatbot allows users to upload documents and ask questions in natural language. 
It retrieves the most relevant information using semantic embeddings and generates precise, source-aware answers using LLMs such as OpenAI, Groq, or OpenRouter.

🔹 Multi-format support.

🔹 Semantic search using embeddings

🔹 Source document referencing

🔹 API-switchable LLM integration

🔹 Session memory & confidence scoring

🔹 Real-time web UI + export features

🚀 Live Demo
🔗 Web App: https://ragchatbotassignment.streamlit.app

📦 GitHub Repo: https://github.com/Omkar4140/RAG_Chatbot

🧠 Architecture

User Query ─► Embed ─► Retrieve Top-k Chunks ─► Prompt LLM ─► Generate Answer ─► Show Sources
                   

🗂️ Features

✅ Upload PDFs, DOCX, TXT, CSV files

✅ Chunking with overlap (preserves context)

✅ TF-IDF embeddings (upgradable to Sentence-BERT)

✅ FAISS-based vector store

✅ LLMs: OpenAI (GPT), Groq, OpenRouter (Claude, LLaMA)

✅ Streamlit UI with:

Chat interface

Configuration panel

Source references


🛠️ Setup Instructions

🔧 1. Clone the Repo

git clone https://github.com/Omkar4140/RAG_Chatbot.git

cd RAG_Chatbot

📦 2. Create Environment & Install Dependencies

pip install -r requirements.txt

🔑 3. Add API Keys
Create a .streamlit/secrets.toml file:

GROQ_API_KEY = "your_groq_key"

OPENAI_API_KEY = "your_openai_key"

OPENROUTER_API_KEY = "your_openrouter_key"

▶️ 4. Run the App

streamlit run app.py

🧪 Sample Use Cases

Query	Expected Response
"What is the leave policy?"	Shows leave info from HR PDF

"How to request IT help?"	Gives support contact from IT SOP

"When is the deadline for appraisal?"	Returns from HR calendar file

"What are the rules for remote work?"	Extracts policy document clause


🧩 Future Enhancements

 Sentence-BERT or OpenAI embeddings

 ChromaDB / PGVector for persistent scalable storage

 Highlight matched content in document

 User login and access control

 Multilingual document/question support

 Streamlit Lite mobile optimization



