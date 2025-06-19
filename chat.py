import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re

# Configure page
st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide"
)

# Load API key from secrets
@st.cache_resource
def initialize_llm():
    try:
        llm = ChatOpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            temperature=0.3
        )
        return llm
    except KeyError:
        st.error("❌ API key missing! Please add OPENAI_API_KEY to Streamlit Secrets.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        st.stop()

def clean_text(text):
    """Clean and preprocess the uploaded text"""
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_chunk_size=4000):
    """Split text into chunks if it's too long"""
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk + paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_answer(llm, document_text, question):
    """Generate answer based on document content and user question"""
    
    # Create system prompt
    system_prompt = """You are a helpful document analysis assistant. Your task is to answer questions based ONLY on the provided document content.

Guidelines:
- Answer questions using only information from the provided document
- If the answer is not in the document, clearly state "The document doesn't contain information about this topic"
- Be precise and cite relevant parts of the document when possible
- If the question is unclear, ask for clarification
- Provide detailed answers when the information is available in the document"""

    # Create user prompt with document content
    user_prompt = f"""Document Content:
---
{document_text}
---

Question: {question}

Please answer the question based on the document content above."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    # Custom styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📄 Document Q&A Agent</h1>', unsafe_allow_html=True)
    st.markdown("Upload a text document and ask questions about its content!")

    # Initialize LLM
    llm = initialize_llm()

    # Initialize session state
    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a text file", 
            type=['txt', 'md', 'py', 'csv', 'json'],
            help="Upload a text document to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Read file content
                content = uploaded_file.read()
                
                # Try to decode as UTF-8
                try:
                    document_text = content.decode('utf-8')
                except UnicodeDecodeError:
                    document_text = content.decode('latin-1')
                
                # Clean and store document
                document_text = clean_text(document_text)
                st.session_state.document_content = document_text
                st.session_state.document_uploaded = True
                
                # Show document stats
                word_count = len(document_text.split())
                char_count = len(document_text)
                
                st.success("✅ Document uploaded successfully!")
                st.info(f"📊 Document Stats:\n- Characters: {char_count:,}\n- Words: {word_count:,}")
                
                # Show document preview
                st.subheader("📖 Document Preview")
                preview_text = document_text[:500] + "..." if len(document_text) > 500 else document_text
                st.text_area("First 500 characters:", preview_text, height=200, disabled=True)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.session_state.document_uploaded = False
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main content area
    if not st.session_state.document_uploaded:
        st.info("👈 Please upload a text document from the sidebar to get started!")
        
        # Show example of what the agent can do
        st.subheader("🤖 What can this agent do?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Document Analysis:**
            - Summarize the document
            - Extract key points
            - Find specific information
            - Explain concepts mentioned
            """)
        
        with col2:
            st.markdown("""
            **Question Types:**
            - "What is this document about?"
            - "Who are the main characters?"
            - "What are the key findings?"
            - "Explain [specific topic] from the document"
            """)
    
    else:
        # Document is uploaded, show Q&A interface
        st.subheader("💬 Ask questions about your document")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Agent:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
        
        # Question input
        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?",
                key="question_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                submit_button = st.form_submit_button("🚀 Ask", use_container_width=True)
            
            with col2:
                example_button = st.form_submit_button("💡 Example", use_container_width=True)
        
        # Handle form submission
        if submit_button and user_question.strip():
            with st.spinner("🤔 Thinking..."):
                # Generate answer
                answer = generate_answer(llm, st.session_state.document_content, user_question)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, answer))
                
                # Rerun to display new message
                st.rerun()
        
        # Handle example button
        if example_button:
            example_questions = [
                "What is this document about?",
                "Summarize the main points",
                "What are the key topics discussed?",
                "Extract the most important information"
            ]
            
            # Use the first example question
            example_q = example_questions[0]
            with st.spinner("🤔 Thinking..."):
                answer = generate_answer(llm, st.session_state.document_content, example_q)
                st.session_state.chat_history.append((example_q, answer))
                st.rerun()
        
        # Quick action buttons
        if len(st.session_state.chat_history) == 0:
            st.subheader("🚀 Quick Start")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Summarize Document", use_container_width=True):
                    with st.spinner("📝 Creating summary..."):
                        answer = generate_answer(llm, st.session_state.document_content, "Please provide a comprehensive summary of this document, highlighting the main topics and key points.")
                        st.session_state.chat_history.append(("Summarize this document", answer))
                        st.rerun()
            
            with col2:
                if st.button("🔍 Extract Key Points", use_container_width=True):
                    with st.spinner("🔍 Extracting key points..."):
                        answer = generate_answer(llm, st.session_state.document_content, "Extract and list the key points, important facts, and main ideas from this document.")
                        st.session_state.chat_history.append(("Extract key points", answer))
                        st.rerun()

if __name__ == "__main__":
    main()
