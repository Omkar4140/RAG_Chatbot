import streamlit as st
import os
from rag_backend import RAGApplication


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4f8;
        border-left-color: #28a745;
    }
    .source-doc {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


if 'rag_app' not in st.session_state:
    st.session_state.rag_app = RAGApplication()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False


st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Chatbot</h1>
    <p style="color: white; text-align: center; margin: 0;">
        Intelligent Assistant for Internal Knowledge Base
    </p>
</div>
""", unsafe_allow_html=True)


left_col, center_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("ℹ️ About")
    st.write("""
    RAG chatbot for document analysis using multiple AI providers.
    
    **Features:**
    - Multi-format support
    - Source references
    - Multiple AI providers
    """)
    
    st.markdown("---")
    
    st.header("💡 Usage Tips")
    tips = [
        "Upload PDFs, TXT, CSV, DOC files",
        "Ask specific questions",
        "Check source documents",
        "Use different AI models"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    st.markdown("---")
    
    st.header("📊 Information")

    api_provider = getattr(st.session_state.rag_app, 'api_provider', 'Not set')
    
    status_items = [
        ("API Provider", api_provider),
        ("Documents", "✅ Processed" if st.session_state.documents_processed else "❌ Not processed"),
        ("Chat History", f"{len(st.session_state.chat_history)} messages")
    ]
    
    for item, status in status_items:
        st.write(f"**{item}:** {status}")
    
    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    

    if st.button("🔄 Reset Application", use_container_width=True):
        st.session_state.rag_app = RAGApplication()
        st.session_state.chat_history = []
        st.session_state.documents_processed = False
        st.rerun()


with center_col:
    st.header("💬 Chat Interface")
    

    if not hasattr(st.session_state.rag_app, 'llm') or st.session_state.rag_app.llm is None:
        st.info("👉 Please configure API settings and upload documents to start chatting")
    elif not st.session_state.documents_processed:
        st.info("👉 Please upload and process documents to start chatting")
    else:

        for i, (role, message, sources) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
             
                if sources:
                    with st.expander("📄 Source Documents", expanded=False):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {j+1}:</strong><br>
                                {source.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is the main topic of the uploaded documents?",
                key="user_input"
            )
            submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)
        
        
        if submit_button and user_question.strip():
            st.session_state.chat_history.append(("user", user_question, []))
            
            
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_app.get_answer(user_question)
            
            if "error" in response:
                bot_message = f"❌ {response['error']}"
                sources = []
            else:
                bot_message = response["answer"]
                sources = response.get("source_documents", [])
            
            st.session_state.chat_history.append(("bot", bot_message, sources))
            
            st.rerun()


with right_col:
    st.header("🛠️ Configuration")
    

    api_provider = st.selectbox(
        "Choose AI Provider",
        ["Groq", "OpenAI", "OpenRouter"],
        index=0
    )
    

    if api_provider == "Groq":
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.secrets.get("GROQ_API_KEY", ""),
            help="Get free key from console.groq.com"
        )
        model_options = ["mistral-saba-24b", "llama3-70b-8192", "gemma2-9b-it", "meta-llama/llama-4-scout-17b-16e-instruct"]
    elif api_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.secrets.get("OPENAI_API_KEY", ""),
            help="Get key from platform.openai.com"
        )
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    else:  
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.secrets.get("OPENROUTER_API_KEY", ""),
            help="Get key from openrouter.ai"
        )
        model_options = ["openai/gpt-3.5-turbo", "anthropic/claude-2", "meta-llama/llama-2-70b-chat"]
    

    selected_model = st.selectbox("Select Model", model_options)
    

    if api_key and st.button("Configure API", use_container_width=True):
        if st.session_state.rag_app.setup_llm(api_provider, api_key, selected_model):
            st.success(f"✅ {api_provider} configured successfully!")
        else:
            st.error(f"❌ Failed to configure {api_provider}")
    
    st.markdown("---")
    
    st.header("📁 Upload Documents")
    

    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv', 'doc', 'docx'],
        help="Upload your documents"
    )
    
    if uploaded_files and hasattr(st.session_state.rag_app, 'llm') and st.session_state.rag_app.llm:
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                if st.session_state.rag_app.process_documents(uploaded_files):
                    st.session_state.documents_processed = True
                    st.success("✅ Documents processed!")
                else:
                    st.error("❌ Failed to process documents")


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Enterprise RAG Chatbot | Multi-Provider AI Support</p>
</div>
""", unsafe_allow_html=True)
