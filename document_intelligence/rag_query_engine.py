# rag_query_engine.py (Fixed Error)
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import httpx

# Import our modules
from embedding_manager import EmbeddingManager, EmbeddingConfig, VectorDBConfig

class Source(BaseModel):
    """Source information for a document"""
    title: str
    url: str
    heading: str = ""  # Default empty string instead of None

class QueryResult(BaseModel):
    """Result of a query against the documentation"""
    query: str
    answer: str
    context_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)

class RAGQueryEngine:
    """Retrieval-Augmented Generation engine using vector database and LLM"""
    
    def __init__(self,
                embedding_manager: Optional[EmbeddingManager] = None,
                embedding_config: Optional[Dict] = None,
                vectordb_config: Optional[Dict] = None,
                openrouter_api_key: Optional[str] = None,
                model: str = "deepseek/deepseek-chat-v3-0324:free"):
        """
        Initialize the RAG query engine
        
        Args:
            embedding_manager: Initialized EmbeddingManager (will create new one if None)
            embedding_config: Configuration for embedding generation (if creating new manager)
            vectordb_config: Configuration for vector database (if creating new manager)
            openrouter_api_key: API key for OpenRouter (or from OPENROUTER_API_KEY env var)
            model: Model to use for generation
        """
        # Initialize or use provided embedding manager
        if embedding_manager:
            self.embedding_manager = embedding_manager
        else:
            self.embedding_manager = EmbeddingManager(
                embedding_config=embedding_config,
                vectordb_config=vectordb_config
            )
        
        # Set OpenRouter API key and model
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
    
    async def query(self, query: str, top_k: int = 5) -> QueryResult:
        """
        Query the documentation using RAG
        
        Args:
            query: User query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            QueryResult with answer and sources
        """
        # Retrieve relevant chunks from vector database
        relevant_chunks = self.embedding_manager.similar_search(query, k=top_k)
        
        # Extract source information
        sources = []
        for chunk in relevant_chunks:
            # Set default empty string for heading if None
            heading = chunk["metadata"].get("heading", "") or ""
            
            source = Source(
                title=chunk["metadata"].get("title", "Unknown"),
                url=chunk["metadata"].get("url", "Unknown"),
                heading=heading
            )
            
            # Deduplicate sources
            if not any(s.url == source.url and s.heading == source.heading for s in sources):
                sources.append(source)
        
        # Format context for the LLM
        context_texts = []
        for i, chunk in enumerate(relevant_chunks):
            heading_info = ""
            chunk_heading = chunk["metadata"].get("heading", "")
            if chunk_heading:
                heading_info = f" - {chunk_heading}"
            
            context_texts.append(
                f"[Document {i+1}{heading_info}]\n{chunk['content']}\n"
            )
        
        context = "\n\n".join(context_texts)
        
        # Generate answer using LLM via OpenRouter API
        answer = await self._generate_answer(query, context)
        
        # Create and return result
        return QueryResult(
            query=query,
            answer=answer,
            context_chunks=relevant_chunks,
            sources=sources
        )
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using OpenRouter API"""
        if not self.openrouter_api_key:
            return "Error: OpenRouter API key not provided. Please set the OPENROUTER_API_KEY environment variable."
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create a prompt that encourages citation and proper use of context
        system_prompt = """You are a helpful documentation assistant. Answer the user's question based on the provided documentation context. 
Be concise, accurate, and cite your sources by referring to the document numbers in your answer (e.g., "According to [Document 1]...").
If the context doesn't contain the information needed, acknowledge that you don't have enough information to answer accurately."""
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocumentation Context:\n{context}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

async def main():
    """Example usage of the RAG query engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Documentation RAG Query Engine")
    parser.add_argument("--vectordb-dir", required=True, help="Path to vector database directory")
    parser.add_argument("--query", required=True, help="Query to run against the documentation")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3-0324:free", help="Model to use for generation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Configure vector database
    vectordb_config = VectorDBConfig(persist_directory=args.vectordb_dir)
    
    # Initialize embedding manager with existing vector DB
    embedding_manager = EmbeddingManager(vectordb_config=vectordb_config)
    
    # Initialize RAG query engine
    engine = RAGQueryEngine(
        embedding_manager=embedding_manager,
        openrouter_api_key=args.api_key,
        model=args.model
    )
    
    print(f"Running query: {args.query}")
    result = await engine.query(args.query, top_k=args.top_k)
    
    print("\n--- Answer ---")
    print(result.answer)
    
    print("\n--- Sources ---")
    for source in result.sources:
        heading_info = f" - {source.heading}" if source.heading else ""
        print(f"- {source.title}{heading_info} ({source.url})")

if __name__ == "__main__":
    asyncio.run(main())