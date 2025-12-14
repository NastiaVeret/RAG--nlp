import streamlit as st
import os
from src.rag.retriever import Retriever
from src.rag.reranker import ReRanker
from src.rag.llm_service import LLMService
from src.rag.metadata_filter_generator import MetadataFilterGenerator

def configure_page():
    st.set_page_config(
        page_title="RAG Knowledge Base",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Reset & Dark Theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    /* Header & Top Elements Cleanup - RESTORED DEFAULT */
    /* 
    header {
        visibility: hidden !important;
    }
    */
    .stDeployButton {
        display: none;
    }
    /*
    #MainMenu {
        visibility: hidden;
    }
    */

    /* Text & Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        background: linear-gradient(120deg, #5EEAD4, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        padding-bottom: 1rem;
    }
    p, .stMarkdown, .stText {
        color: #b0b8c4 !important;
        line-height: 1.6;
    }

    /* Input Fields styling - High Contrast */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-testid="stMarkdownContainer"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 8px;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #6e7681 !important;
    }
    /* Focus states */
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: transparent;
        border-bottom: 1px solid #30363d;
    }
    div[data-testid="stChatMessageContent"] {
        background: transparent !important;
        color: #e0e0e0 !important;
    }

    /* Premium Source Cards */
    .source-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(94, 234, 212, 0.3);
    }
    .source-title {
        color: #5EEAD4;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 4px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(to right, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.025em;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: linear-gradient(to right, #2563eb, #1d4ed8);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Warning/Info/Success Messages */
    .stAlert {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components(api_key):
    retriever = Retriever()
    reranker = ReRanker()
    llm_service = LLMService(api_key=api_key)
    filter_generator = MetadataFilterGenerator(api_key=api_key)
    return retriever, reranker, llm_service, filter_generator

def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "sources" in message:
                st.markdown("### Sources")
                cols = st.columns(len(message["sources"]))
                for i, source in enumerate(message["sources"]):
                    with cols[i]:
                         st.markdown(f"""
<div class="source-card">
    <div class="source-title">Source {i+1}</div>
    {source[:300]}...
</div>
""", unsafe_allow_html=True)
                
                st.markdown("### Answer")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

def process_query(prompt, retriever, reranker, llm_service, filter_generator):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.status("Processing request...", expanded=True) as status:
            st.write("Standard Dense Retrieval...")
            metadata_filter = None
            initial_results = retriever.search_semantic(
                prompt,
                top_k=10,
                metadata_filter=None
            )
            if not initial_results:
                st.write("No results found. analyzing query for metadata filters...")
                metadata_filter = filter_generator.generate_filter(prompt)
                
                if metadata_filter:
                    filter_explanation = filter_generator.explain_filter(metadata_filter)
                    st.write(f"Filter applied: {filter_explanation}")
                    
                    st.write("Retrying with Metadata Filtering...")
                    initial_results = retriever.search_semantic(
                        prompt, 
                        top_k=10,
                        metadata_filter=metadata_filter
                    )
                else:
                    st.write("No applicable metadata filters found.")
                
            st.write(f"Retrieved {len(initial_results)} candidates")
            
            st.write("Re-ranking candidates...")
            ranked_results = reranker.rerank(prompt, initial_results, top_k=3)
            st.write(f"Selected top {len(ranked_results)} matches")
            
            st.write("Generating answer...")
            
            status.update(label="Complete!", state="complete", expanded=False)

        context_chunks = [res['chunk'] for res in ranked_results]
        
        response_data = llm_service.generate_response(prompt, context_chunks)
        generated_answer = response_data["answer"]
        sources_text = response_data["sources"]

        if sources_text:
            st.markdown("### Sources")
            cols = st.columns(len(sources_text))
            for i, source in enumerate(sources_text):
                with cols[i]:
                    st.markdown(f"""
<div class="source-card">
    <div class="source-title">Source {i+1}</div>
    {source[:300]}... 
</div>
""", unsafe_allow_html=True)

        st.markdown("### Answer")
        st.markdown(generated_answer)
        
        if metadata_filter:
            st.markdown("---")
            st.markdown(f"**Applied Filters:** {filter_generator.explain_filter(metadata_filter)}")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": generated_answer,
            "sources": sources_text,
            "filter": metadata_filter
        })

def main():
    configure_page()
    inject_custom_css()
    
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API Key here")
    
    st.title("Intelligent Knowledge Base")
    st.markdown("""
### Welcome!
This service uses advanced **RAG (Retriever -> Re-Ranker -> Generator)** architecture with **Metadata Filtering**.
""")

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to continue.")
        return

    try:
        with st.spinner("Loading AI components..."):
            retriever, reranker, llm_service, filter_generator = load_components(api_key)
    except Exception as e:
        st.error(f"Critical error during loading: {str(e)}")
        st.stop()

    display_chat_history()

    if prompt := st.chat_input("What would you like to know?"):
        process_query(prompt, retriever, reranker, llm_service, filter_generator)

if __name__ == "__main__":
    main()
