# document_processor.py
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain imports for document processing - Updated import paths
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

class DocumentChunk(BaseModel):
    """A chunk of documentation content with metadata"""
    chunk_id: str
    title: str
    content: str
    url: str
    heading: Optional[str] = None
    parent_heading: Optional[str] = None
    depth: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_langchain_document(self) -> LangchainDocument:
        """Convert to LangChain Document format"""
        metadata = {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "heading": self.heading,
            "parent_heading": self.parent_heading,
            "depth": self.depth,
            **self.metadata
        }
        
        return LangchainDocument(page_content=self.content, metadata=metadata)

def extract_headings(html_content: str) -> List[str]:
    """Extract heading information from HTML content"""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = []
    
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        headings.append({
            'level': int(tag.name[1]),
            'text': tag.get_text(strip=True),
            'id': tag.get('id', '')
        })
    
    return headings

def process_crawl_results(crawl_file: str, output_dir: str, 
                         chunk_size: int = 1000, 
                         chunk_overlap: int = 200) -> str:
    """
    Process crawled documentation, extract metadata, and create chunks
    
    Args:
        crawl_file: Path to the crawl results JSON file
        output_dir: Directory to save processed chunks
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Path to the processed chunks file
    """
    # Load crawled documentation
    with open(crawl_file, 'r', encoding='utf-8') as f:
        crawl_data = json.load(f)
    
    # Extract pages
    pages = crawl_data.get('pages', [])
    
    if not pages:
        raise ValueError("No pages found in crawl results")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process each page and create chunks
    all_chunks = []
    chunk_id_counter = 0
    
    for page_idx, page in enumerate(pages):
        url = page.get('url', '')
        title = page.get('title', f'Page {page_idx}')
        content = page.get('content', '')
        html_content = page.get('html_content', '')
        depth = page.get('depth', 0)
        
        # Extract headings if HTML content is available
        headings = []
        if html_content:
            headings = extract_headings(html_content)
        
        # Create metadata
        metadata = {
            "source": url,
            "title": title,
            "depth": depth,
            "headings": headings
        }
        
        # Create LangChain document
        doc = LangchainDocument(page_content=content, metadata=metadata)
        
        # Split into chunks
        chunks = text_splitter.split_documents([doc])
        
        # Convert to DocumentChunk format
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{chunk_id_counter}"
            chunk_id_counter += 1
            
            # Try to determine the heading context for this chunk
            current_heading = None
            parent_heading = None
            
            # If we have heading information, try to find the relevant heading
            # for this chunk based on position in text (simplified approach)
            if headings:
                # This is a simplified approach - in a real implementation,
                # you'd need more sophisticated logic to determine which heading
                # each chunk belongs to
                if len(chunks) > 1:
                    # Determine which section of the document this chunk represents
                    chunk_position = i / len(chunks)
                    heading_idx = int(chunk_position * len(headings))
                    if heading_idx < len(headings):
                        current_heading = headings[heading_idx]['text']
                        # If there's a parent heading (based on heading level)
                        if heading_idx > 0 and headings[heading_idx]['level'] > headings[0]['level']:
                            for h in reversed(headings[:heading_idx]):
                                if h['level'] < headings[heading_idx]['level']:
                                    parent_heading = h['text']
                                    break
                else:
                    # If only one chunk, use the first heading
                    if headings:
                        current_heading = headings[0]['text']
            
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                title=title,
                content=chunk.page_content,
                url=url,
                heading=current_heading,
                parent_heading=parent_heading,
                depth=depth,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_page_idx": page_idx
                }
            )
            
            all_chunks.append(doc_chunk)
    
    # Save chunks to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_chunks.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([chunk.model_dump() for chunk in all_chunks], f, indent=2)
    
    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    print(f"Saved processed chunks to {output_file}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process crawled documentation")
    parser.add_argument("--crawl-file", required=True, help="Path to crawl results JSON file")
    parser.add_argument("--output-dir", default="./data", help="Directory to save processed chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    output_file = process_crawl_results(
        crawl_file=args.crawl_file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print(f"Document processing complete! Chunks saved to: {output_file}")