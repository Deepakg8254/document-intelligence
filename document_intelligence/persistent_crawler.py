# persistent_crawler.py
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from pydantic import BaseModel
from urllib.parse import urljoin, urlparse

class PageResult(BaseModel):
    """Individual page result from crawling"""
    url: str
    title: str
    content: str
    depth: int
    parent_url: Optional[str] = None
    crawl_time: str

class DeepCrawlResults(BaseModel):
    """Collection of results from deep crawling"""
    base_url: str
    pages: List[PageResult]
    start_time: str
    end_time: str
    total_pages: int
    max_depth_reached: int

async def persistent_crawl(
    start_url: str, 
    output_dir: str,
    max_pages: int = 20,
    max_depth: int = 2
) -> str:
    """
    Crawl documentation with a persistent browser session.
    
    Args:
        start_url: Starting URL for the documentation
        output_dir: Directory to save the results
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth to crawl
    
    Returns:
        Path to the JSON file containing the crawl results
    """
    try:
        from crawl4ai import AsyncWebCrawler
        from bs4 import BeautifulSoup
        import re
    except ImportError as e:
        raise ImportError(f"Required library not installed: {e}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    start_time = datetime.now()
    start_time_str = start_time.isoformat()
    
    # Parse base domain
    parsed_url = urlparse(start_url)
    base_domain = parsed_url.netloc
    
    # Keep track of crawled and queued URLs
    crawled_urls = set()
    to_crawl = [(start_url, 0, None)]  # (url, depth, parent_url)
    
    # Results
    pages = []
    max_depth_reached = 0
    
    print(f"Starting crawl from {start_url}...")
    print(f"Will crawl up to {max_pages} pages to a maximum depth of {max_depth}")
    
    # Create a single browser instance to use for all requests
    async with AsyncWebCrawler() as crawler:
        page_count = 0
        
        while to_crawl and page_count < max_pages:
            current_url, depth, parent_url = to_crawl.pop(0)
            
            if current_url in crawled_urls:
                continue
                
            if depth > max_depth:
                continue
                
            max_depth_reached = max(max_depth_reached, depth)
            
            print(f"Crawling ({page_count+1}/{max_pages}) [Depth {depth}/{max_depth}]: {current_url}")
            
            try:
                # Crawl the current page
                result = await crawler.arun(url=current_url)
                
                # Extract content based on what's available
                content = ""
                html_content = ""
                
                if hasattr(result, 'markdown'):
                    content = result.markdown
                elif hasattr(result, 'text'):
                    content = result.text
                else:
                    content = str(result)
                    
                if hasattr(result, 'html'):
                    html_content = result.html
                
                # Create page result
                page = PageResult(
                    url=current_url,
                    title=getattr(result, 'title', f"Page {page_count+1}"),
                    content=content,
                    depth=depth,
                    parent_url=parent_url,
                    crawl_time=datetime.now().isoformat()
                )
                
                pages.append(page)
                crawled_urls.add(current_url)
                page_count += 1
                
                # Only extract links if we haven't reached max depth
                if depth < max_depth and html_content:
                    # Parse HTML and extract links
                    soup = BeautifulSoup(html_content, 'html.parser')
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        full_url = urljoin(current_url, href)
                        parsed = urlparse(full_url)
                        
                        # Only include links to the same domain and ignore fragments
                        if (parsed.netloc == base_domain and 
                            '#' not in full_url and 
                            not full_url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf'))):
                            
                            # Clean the URL (remove query params if needed)
                            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            
                            if clean_url not in crawled_urls and not any(clean_url == u[0] for u in to_crawl):
                                to_crawl.append((clean_url, depth + 1, current_url))
            
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                crawled_urls.add(current_url)
    
    # End time
    end_time = datetime.now()
    end_time_str = end_time.isoformat()
    
    # Create results object
    crawl_results = DeepCrawlResults(
        base_url=start_url,
        pages=pages,
        start_time=start_time_str,
        end_time=end_time_str,
        total_pages=len(pages),
        max_depth_reached=max_depth_reached
    )
    
    # Generate filename and save results
    safe_domain = start_url.replace("https://", "").replace("http://", "").split("/")[0]
    safe_domain = "".join(c if c.isalnum() else "_" for c in safe_domain)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{safe_domain}_depth{max_depth}_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(crawl_results.model_dump_json(indent=2))
    
    print(f"\nCrawling complete! Crawled {len(pages)} pages.")
    print(f"Saved results to {file_path}")
    
    return file_path

async def main():
    """Command line interface for the crawler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persistent Browser Documentation Crawler")
    parser.add_argument("--url", required=True, help="URL of the documentation to crawl")
    parser.add_argument("--output", default="./data", help="Output directory for crawled documentation")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to crawl")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum crawl depth")
    
    args = parser.parse_args()
    
    result_file = await persistent_crawl(
        start_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
        max_depth=args.max_depth
    )
    
    print(f"Crawl complete! Results saved to: {result_file}")

if __name__ == "__main__":
    asyncio.run(main())