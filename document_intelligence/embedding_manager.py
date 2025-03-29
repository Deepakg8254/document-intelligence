# embedding_manager.py (Simplified)
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

# Pydantic for data validation
from pydantic import BaseModel, Field

# Use sentence_transformers directly instead of LangChain wrappers
from sentence_transformers import SentenceTransformer
import torch

# Import our document processor
from document_processor import DocumentChunk

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512

class VectorDBConfig(BaseModel):
    """Configuration for vector database"""
    persist_directory: str = "./vectordb"
    collection_name: str = "documentation"

class EmbeddingManager:
    """Simplified embedding manager without LangChain dependencies"""
    
    def __init__(self, 
                embedding_config: Optional[Union[EmbeddingConfig, Dict]] = None,
                vectordb_config: Optional[Union[VectorDBConfig, Dict]] = None):
        """
        Initialize the embedding manager
        
        Args:
            embedding_config: Configuration for embedding generation
            vectordb_config: Configuration for vector database
        """
        # Initialize configurations
        self.embedding_config = (
            EmbeddingConfig(**embedding_config) if isinstance(embedding_config, dict) 
            else embedding_config or EmbeddingConfig()
        )
        
        self.vectordb_config = (
            VectorDBConfig(**vectordb_config) if isinstance(vectordb_config, dict) 
            else vectordb_config or VectorDBConfig()
        )
        
        # Initialize embedding model directly with sentence-transformers
        self.model = SentenceTransformer(self.embedding_config.model_name, device=self.embedding_config.device)
        
        # Initialize vector database (in-memory for simplicity)
        self.document_embeddings = []
        self.documents = []
        
        # Create persist directory if it doesn't exist
        Path(self.vectordb_config.persist_directory).mkdir(parents=True, exist_ok=True)
        self.db_file = os.path.join(self.vectordb_config.persist_directory, "embeddings.json")
        
        # Load existing embeddings if available
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing embeddings from file if available"""
        if os.path.exists(self.db_file):
            print(f"Loading existing embeddings from {self.db_file}")
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.document_embeddings = data.get('embeddings', [])
                print(f"Loaded {len(self.documents)} documents with embeddings")
            except Exception as e:
                print(f"Error loading existing embeddings: {e}")
                self.documents = []
                self.document_embeddings = []
    
    def _save_embeddings(self):
        """Save embeddings to file"""
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'embeddings': self.document_embeddings
            }, f)
        print(f"Saved {len(self.documents)} documents with embeddings to {self.db_file}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            return
        
        # Prepare texts and metadata for embedding
        texts = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "url": chunk.url,
                "heading": chunk.heading,
                "parent_heading": chunk.parent_heading,
                "depth": chunk.depth,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True).tolist()
            
            # Store documents and embeddings
            for j, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadatas, batch_embeddings)):
                document = {
                    "content": text,
                    "metadata": metadata
                }
                self.documents.append(document)
                self.document_embeddings.append(embedding)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Save embeddings to file
        self._save_embeddings()
        
        print(f"Added {len(chunks)} chunks to vector database")
    
    def load_and_add_chunks_from_file(self, chunks_file: str) -> None:
        """
        Load chunks from a file and add them to the vector database
        
        Args:
            chunks_file: Path to the JSON file containing document chunks
        """
        # Load chunks from file
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        # Convert to DocumentChunk objects
        chunks = [DocumentChunk(**chunk) for chunk in chunk_data]
        
        # Add to vector database
        self.add_documents(chunks)
        
        print(f"Loaded and added {len(chunks)} chunks from {chunks_file}")
    
    def similar_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = sum(a*b for a, b in zip(query_embedding, doc_embedding))
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Return top k results
        results = []
        for similarity, idx in similarities[:k]:
            document = self.documents[idx]
            result = {
                "content": document["content"],
                "metadata": document["metadata"],
                "similarity": similarity
            }
            results.append(result)
        
        return results

def create_vector_db_from_chunks(chunks_file: str, 
                                persist_directory: str = "./vectordb",
                                model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    Create a vector database from document chunks
    
    Args:
        chunks_file: Path to the JSON file containing document chunks
        persist_directory: Directory to persist the vector database
        model_name: Name of the embedding model to use
        
    Returns:
        Initialized EmbeddingManager
    """
    # Create configuration
    embedding_config = EmbeddingConfig(model_name=model_name)
    vectordb_config = VectorDBConfig(persist_directory=persist_directory)
    
    # Initialize embedding manager
    manager = EmbeddingManager(
        embedding_config=embedding_config,
        vectordb_config=vectordb_config
    )
    
    # Load and add chunks
    manager.load_and_add_chunks_from_file(chunks_file)
    
    return manager

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create vector database from document chunks")
    parser.add_argument("--chunks-file", required=True, help="Path to processed chunks JSON file")
    parser.add_argument("--persist-dir", default="./vectordb", help="Directory to persist the vector database")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--test-query", help="Optional query to test the vector database")
    
    args = parser.parse_args()
    
    # Create vector database
    manager = create_vector_db_from_chunks(
        chunks_file=args.chunks_file,
        persist_directory=args.persist_dir,
        model_name=args.model
    )
    
    print(f"Vector database created and persisted to {args.persist_dir}")
    
    # Test query if provided
    if args.test_query:
        print(f"\nTesting query: '{args.test_query}'")
        results = manager.similar_search(args.test_query, k=3)
        
        print("\nTop results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (Similarity: {result['similarity']:.4f}) ---")
            print(f"Title: {result['metadata'].get('title', 'Unknown')}")
            print(f"URL: {result['metadata'].get('url', 'Unknown')}")
            if result['metadata'].get('heading'):
                print(f"Heading: {result['metadata']['heading']}")
            print(f"\nContent snippet: {result['content'][:200]}...")