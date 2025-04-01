
import streamlit as st
import asyncio
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional
import httpx
from dotenv import load_dotenv

# Import our custom modules
from persistent_crawler import persistent_crawl
from document_processor import process_crawl_results
from embedding_manager import EmbeddingManager, VectorDBConfig, create_vector_db_from_chunks
from rag_query_engine import RAGQueryEngine

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Documentation Intelligence System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "crawl_results" not in st.session_state:
    st.session_state.crawl_results = None
if "crawl_file_path" not in st.session_state:
    st.session_state.crawl_file_path = None
if "chunks_file_path" not in st.session_state:
    st.session_state.chunks_file_path = None
if "vectordb_path" not in st.session_state:
    st.session_state.vectordb_path = None
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None
if "crawling" not in st.session_state:
    st.session_state.crawling = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = bool(os.environ.get("OPENROUTER_API_KEY"))

# Function to run the crawler asynchronously
async def run_crawler(url: str, max_pages: int, max_depth: int) -> str:
    """Run the crawler and return the path to the results file"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    return await persistent_crawl(
        start_url=url,
        output_dir=str(data_dir),
        max_pages=max_pages,
        max_depth=max_depth
    )

# Header
st.title("📚 Documentation Intelligence System")
st.markdown("""
This app crawls technical documentation, processes it with advanced NLP techniques, and answers your questions using vector embeddings and LLMs.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenRouter API Key", type="password", 
                        help="Get one from openrouter.ai. Required for queries.")
    
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        st.session_state.api_key_set = True
    
    # Crawler Configuration
    st.subheader("Crawler Settings")
    doc_url = st.text_input("Documentation URL", "https://fastapi.tiangolo.com/")
    max_pages = st.slider("Maximum Pages", min_value=1, max_value=30, value=10)
    max_depth = st.slider("Maximum Depth", min_value=1, max_value=3, value=2)
    
    # Document Processing
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=800)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150)
    
    # Embedding Model
    st.subheader("Embedding Model")
    embedding_model = st.selectbox(
        "Model",
        ["sentence-transformers/all-MiniLM-L6-v2", 
         "sentence-transformers/multi-qa-mpnet-base-dot-v1"],
    )
    
    # LLM Model
    st.subheader("LLM Model")
    llm_model = st.selectbox(
        "Model",
        ["deepseek/deepseek-chat-v3-0324:free", "anthropic/claude-3-haiku-20240307"],
    )
    
    # Action buttons
    crawl_col, process_col = st.columns(2)
    
    with crawl_col:
        if st.button("Crawl"):
            if not doc_url:
                st.error("Please enter a URL")
            else:
                st.session_state.crawling = True
                st.session_state.crawl_results = None
                st.session_state.crawl_file_path = None
    
    # Process existing crawl data
    with process_col:
        if st.button("Process"):
            if not st.session_state.crawl_file_path:
                st.error("No crawl data. Please crawl first.")
            else:
                st.session_state.processing = True

# Main content area - Crawler
if st.session_state.crawling:
    st.subheader("Crawling Documentation")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing crawler...")
    
    crawl_result_path = asyncio.run(run_crawler(doc_url, max_pages, max_depth))
    progress_bar.progress(100)
    
    # Update session state
    st.session_state.crawl_file_path = crawl_result_path
    
    # Load the crawl results
    with open(crawl_result_path, 'r', encoding='utf-8') as f:
        st.session_state.crawl_results = json.load(f)
    
    status_text.text("Crawling complete!")
    st.session_state.crawling = False
    st.experimental_rerun()

# Main content area - Document Processing
if st.session_state.processing:
    st.subheader("Processing Documents")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Processing documents...")
    
    try:
        # Process crawled data and create chunks
        chunks_file = process_crawl_results(
            crawl_file=st.session_state.crawl_file_path,
            output_dir="./data",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        st.session_state.chunks_file_path = chunks_file
        progress_bar.progress(50)
        status_text.text("Creating vector database...")
        
        # Create vector database from chunks
        vectordb_dir = "./vectordb"
        embedding_manager = create_vector_db_from_chunks(
            chunks_file=chunks_file,
            persist_directory=vectordb_dir,
            model_name=embedding_model
        )
        
        st.session_state.vectordb_path = vectordb_dir
        st.session_state.embedding_manager = embedding_manager
        
        progress_bar.progress(100)
        status_text.text("Processing complete! Ready for queries.")
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        progress_bar.progress(100)
        status_text.text("Error during processing. See details above.")
    
    st.session_state.processing = False
    st.experimental_rerun()

# Main content area - Query Interface
if st.session_state.embedding_manager and not st.session_state.crawling and not st.session_state.processing:
    # Display stats
    st.subheader("Documentation Stats")
    
    col1, col2, col3 = st.columns(3)
    
    # Load crawl results
    crawl_data = st.session_state.crawl_results
    col1.metric("Pages Crawled", crawl_data.get("total_pages", 0))
    
    # Load chunks info
    with open(st.session_state.chunks_file_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    col2.metric("Document Chunks", len(chunks_data))
    col3.metric("Vector DB", "Ready ✓")
    
    # Query interface
    st.subheader("Ask a Question")
    
    # Check if API key is set
    if not st.session_state.api_key_set:
        st.warning("Please enter an OpenRouter API key in the sidebar to ask questions.")
    
    query = st.text_input("Your question about the documentation:", "What is this framework about?")
    k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    
    if st.button("Ask") and st.session_state.api_key_set:
        with st.spinner("Generating answer..."):
            try:
                # Initialize RAG query engine
                rag_engine = RAGQueryEngine(
                    embedding_manager=st.session_state.embedding_manager,
                    openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
                    model=llm_model
                )
                
                # Query the documentation
                result = asyncio.run(rag_engine.query(query, top_k=k))
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(result.answer)
                
                # Display sources
                st.markdown("### Sources")
                for i, source in enumerate(result.sources):
                    heading_info = f" - {source.heading}" if source.heading else ""
                    st.markdown(f"**{i+1}. {source.title}{heading_info}**")
                    st.markdown(f"URL: {source.url}")
                
                # Display retrieved chunks in expander
                with st.expander("View retrieved chunks"):
                    for i, chunk in enumerate(result.context_chunks):
                        st.markdown(f"#### Chunk {i+1} (Similarity: {chunk['similarity']:.4f})")
                        st.markdown(f"**Title:** {chunk['metadata'].get('title', 'Unknown')}")
                        if chunk['metadata'].get('heading'):
                            st.markdown(f"**Heading:** {chunk['metadata']['heading'] or ''}")
                        st.markdown(f"**URL:** {chunk['metadata'].get('url', 'Unknown')}")
                        st.text(chunk['content'])
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.error("Please try again with a different question or check the error details.")
                import traceback
                st.expander("Error details").code(traceback.format_exc())
    
# Show crawl results if available but not processed
elif st.session_state.crawl_file_path and not st.session_state.embedding_manager:
    crawl_data = st.session_state.crawl_results
    
    st.subheader("Crawl Results")
    st.metric("Pages Crawled", crawl_data.get("total_pages", 0))
    st.metric("Max Depth", crawl_data.get("max_depth_reached", 0))
    
    st.info("Click 'Process' in the sidebar to create document chunks and build the vector database.")
    
    # Show sample of crawled pages
    with st.expander("View crawled pages"):
        for i, page in enumerate(crawl_data.get("pages", [])[:5]):
            st.markdown(f"### {i+1}. {page.get('title', 'Untitled')}")
            st.markdown(f"URL: {page.get('url', 'Unknown')}")
            st.markdown(f"Depth: {page.get('depth', 0)}")
            with st.expander("View content"):
                st.text(page.get("content", "No content")[:500] + "...")

# First-time instructions
elif not st.session_state.crawl_file_path:
    st.info("👈 Start by entering a documentation URL and clicking 'Crawl' in the sidebar.")
    
    # Architecture explanation
    with st.expander("How it works"):
        st.markdown("""
        ### System Architecture
        
        This documentation intelligence system uses a modern RAG (Retrieval-Augmented Generation) approach:
        
        1. **Web Crawling**: Crawls documentation websites to extract content
        2. **Document Processing**: Chunks documents into smaller units
        3. **Vector Database**: Stores document chunks as vector embeddings
        4. **Semantic Search**: Retrieves the most relevant chunks for a given query
        5. **LLM Integration**: Generates accurate answers based on the retrieved context
        """)
    
    st.markdown("### Example Questions")
    st.markdown("""
    After crawling and processing the documentation, you'll be able to ask questions like:
    
    - What are the key features of this framework?
    - How do I handle file uploads?
    - What's the recommended way to implement authentication?
    - How do I deploy this framework to production?
    - Can you explain how dependency injection works?
    - What are the best practices for error handling?
    """)
    
    # Show workflow
    st.subheader("Workflow")
    st.markdown("""
    1. **Crawl**: Enter a documentation URL and click "Crawl" to collect the data
    2. **Process**: Click "Process" to chunk the documents and build the vector database
    3. **Query**: Ask questions about the documentation and get AI-generated answers
    """)
    
    # Tips
    with st.expander("Tips"):
        st.markdown("""
        - For best results, choose documentation sites with clear structure
        - Adjust chunk size based on the complexity of the documentation
        - Use smaller chunks (400-600) for technical documentation with dense information
        - Use larger chunks (800-1200) for narrative documentation
        - Increase the number of chunks retrieved (k) for complex questions
        - Try different embedding models for different types of documentation
        """)
