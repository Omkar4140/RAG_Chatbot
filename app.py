import streamlit as st
import os
from rag_backend import RAGApplication, SAMPLE_DOCUMENTS
import tempfile

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Initialize session state
if 'rag_app' not in st.session_state:
    st.session_state.rag_app = RAGApplication()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Enterprise RAG Chatbot</h1>
    <p style="color: white; text-align: center; margin: 0;">
        Intelligent Assistant for Internal Knowledge Base
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key from https://console.groq.com/keys"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Please enter your Groq API Key")
    
    st.markdown("---")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model",
        ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
        help="Select the LLM model for generation"
    )
    
    # Update model if changed
    if hasattr(st.session_state.rag_app, 'llm') and st.session_state.rag_app.llm:
        st.session_state.rag_app.llm.model_name = model_choice
    
    st.markdown("---")
    
    # Document upload section
    st.header("📁 Document Upload")
    
    # Sample documents option
    use_sample = st.checkbox("Use Sample Documents", help="Use pre-loaded sample company documents")
    
    if use_sample:
        if st.button("Load Sample Documents"):
            if groq_api_key:
                with st.spinner("Processing sample documents..."):
                    # Create temporary files from sample documents
                    temp_files = []
                    for title, content in SAMPLE_DOCUMENTS.items():
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                        temp_file.write(content)
                        temp_file.close()
                        temp_files.append(temp_file.name)
                    
                    # Process documents
                    class MockFile:
                        def __init__(self, name, content):
                            self.name = name
                            self.content = content
                        def getvalue(self):
                            return self.content.encode()
                    
                    mock_files = [MockFile(f"{title}.txt", content) for title, content in SAMPLE_DOCUMENTS.items()]
                    
                    if st.session_state.rag_app.process_documents(mock_files):
                        st.session_state.documents_processed = True
                        st.success("✅ Sample documents processed successfully!")
                    else:
                        st.error("❌ Failed to process sample documents")
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
            else:
                st.error("Please enter your Groq API Key first")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv', 'doc', 'docx'],
        help="Upload your internal documents (PDF, TXT, CSV, DOC, DOCX)"
    )
    
    if uploaded_files and groq_api_key:
        if st.button("Process Documents"):
            if st.session_state.rag_app.process_documents(uploaded_files):
                st.session_state.documents_processed = True
                st.success("✅ Documents processed successfully!")
            else:
                st.error("❌ Failed to process documents")
    
    st.markdown("---")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Reset application
    if st.button("🔄 Reset Application"):
        st.session_state.rag_app = RAGApplication()
        st.session_state.chat_history = []
        st.session_state.documents_processed = False
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Check if ready to chat
    if not groq_api_key:
        st.info("👈 Please enter your Groq API Key in the sidebar to get started")
    elif not st.session_state.documents_processed:
        st.info("👈 Please upload and process documents or use sample documents to start chatting")
    else:
        # Chat interface
        # Display chat history
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
                
                # Show sources if available
                if sources:
                    with st.expander("📄 Source Documents", expanded=False):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {j+1}:</strong><br>
                                {source.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What is the company's leave policy?",
            key="user_input"
        )
        
        if st.button("Send", type="primary") or user_question:
            if user_question.strip():
                # Add user message to history
                st.session_state.chat_history.append(("user", user_question, []))
                
                # Get response
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.rag_app.get_answer(user_question)
                
                if "error" in response:
                    bot_message = f"❌ {response['error']}"
                    sources = []
                else:
                    bot_message = response["answer"]
                    sources = response.get("source_documents", [])
                
                # Add bot response to history
                st.session_state.chat_history.append(("bot", bot_message, sources))
                
                # Clear input and rerun
                st.rerun()

with col2:
    st.header("📊 Information")
    
    # System status
    st.subheader("System Status")
    
    status_items = [
        ("API Key", "✅ Configured" if groq_api_key else "❌ Not configured"),
        ("Documents", "✅ Processed" if st.session_state.documents_processed else "❌ Not processed"),
        ("Model", f"🤖 {model_choice}" if groq_api_key else "❌ Not available"),
        ("Chat History", f"💬 {len(st.session_state.chat_history)} messages")
    ]
    
    for item, status in status_items:
        st.write(f"**{item}:** {status}")
    
    st.markdown("---")
    
    # Usage tips
    st.subheader("💡 Usage Tips")
    tips = [
        "Upload multiple documents for better coverage",
        "Ask specific questions for better answers",
        "Use sample documents to test the system",
        "Check source documents for context",
        "Clear chat history to start fresh"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.write("""
    This RAG (Retrieval-Augmented Generation) chatbot helps you find information 
    from your internal documents quickly and accurately.
    
    **Features:**
    - Multi-format document support
    - Contextual compression
    - Source document references
    - Free Groq API integration
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Enterprise RAG Chatbot | Built with Streamlit & Groq</p>
</div>
""", unsafe_allow_html=True)