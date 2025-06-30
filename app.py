import streamlit as st
import os
from rag_backend import RAGApplication

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
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
    <h1>🤖 RAG Chatbot</h1>
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
    
    # Document upload section
    st.header("📁 Document Upload")
    
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
        st.info("👈 Please upload and process documents to start chatting")
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
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is the main topic of the uploaded documents?",
                key="user_input"
            )
            submit_button = st.form_submit_button("Send", type="primary")
        
        # Process the query when form is submitted
        if submit_button and user_question.strip():
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
            
            # Rerun to display the new messages
            st.rerun()

with col2:
    st.header("📊 Information")
    
    # System status
    st.subheader("System Status")
    
    status_items = [
        ("API Key", "✅ Configured" if groq_api_key else "❌ Not configured"),
        ("Documents", "✅ Processed" if st.session_state.documents_processed else "❌ Not processed"),
        ("Model", "🤖 Mixtral-8x7B" if groq_api_key else "❌ Not available"),
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
        "Check source documents for context",
        "Clear chat history to start fresh",
        "Supported formats: PDF, TXT, CSV, DOC, DOCX"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.write("""
    This RAG (Retrieval-Augmented Generation) chatbot helps you find information 
    from your uploaded documents quickly and accurately.
    
    **Features:**
    - Multi-format document support
    - Contextual compression
    - Source document references
    - Mixtral-8x7B model via Groq API
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Enterprise RAG Chatbot | Built with Streamlit & Groq</p>
</div>
""", unsafe_allow_html=True)
