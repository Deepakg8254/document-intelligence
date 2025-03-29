# Documentation Intelligence System

A complete RAG (Retrieval-Augmented Generation) system that crawls documentation websites, processes the content, and answers questions using vector search and LLMs.

## License

MIT License

Copyright (c) 2025 Deepak Grover

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Deepakg8254/documentation-intelligence.git
cd documentation-intelligence
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create environment file

Create a `.env` file in the project root with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

### 5. Create required directories

```bash
mkdir -p data vectordb docs
```

## Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

## Step-by-Step Usage Guide

### Step 1: Crawl a Documentation Site

1. Enter a URL (e.g., `https://fastapi.tiangolo.com/`)
2. Set crawler parameters (pages, depth)
3. Click "Crawl"

### Step 2: Process Documents

1. After crawling completes, click "Process"
2. Set chunking parameters and embedding model
3. Wait for processing to complete

### Step 3: Ask Questions

1. Enter your question
2. Set number of chunks to retrieve (default: 5)
3. Click "Ask"
4. View answer and sources

## File Structure

```
documentation-intelligence/
├── app.py            # Main Streamlit app
├── persistent_crawler.py        # Web crawler
├── document_processor.py        # Document chunking
├── embedding_manager.py         # Vector embeddings
├── rag_query_engine.py          # Query handling
├── requirements.txt             # Dependencies
├── .env                         # API keys (create this)
├── data/                        # Crawled data
└── vectordb/                    # Vector database
```

## Components Explained

### 1. Web Crawler (persistent_crawler.py)

- Crawls documentation websites
- Extracts content, titles, URLs
- Follows links up to specified depth
- Saves results as JSON

### 2. Document Processor (document_processor.py)

- Splits documents into chunks
- Extracts headings and structure
- Preserves metadata
- Creates overlap between chunks
- Saves processed chunks as JSON

### 3. Embedding Manager (embedding_manager.py)

- Converts text to vector embeddings
- Uses SentenceTransformer models
- Stores embeddings and metadata
- Performs semantic similarity search
- Handles batched processing for efficiency

### 4. RAG Query Engine (rag_query_engine.py)

- Processes user queries
- Retrieves relevant chunks
- Formats context for LLM
- Calls OpenRouter API
- Returns answers with sources

### 5. Streamlit App (app.py)

- User interface
- Configuration options
- Step-by-step workflow
- Results visualization

## Technical Details

### Dependencies

- `streamlit`: Web interface
- `crawl4ai`: Web crawling
- `sentence-transformers`: Vector embeddings
- `httpx`: Async HTTP requests
- `pydantic`: Data validation
- `python-dotenv`: Environment variables

### Embedding Models

- Default: `sentence-transformers/all-MiniLM-L6-v2`
- Alternative: `sentence-transformers/multi-qa-mpnet-base-dot-v1`

### LLM Options

- Default: `deepseek/deepseek-chat-v3-0324:free`
- Alternative: `anthropic/claude-3-haiku-20240307`

## Troubleshooting

### Common Issues:

1. **Crawling fails**: Check if site allows crawling
2. **Processing is slow**: Reduce chunks or change model
3. **Out of memory**: Lower chunk size or crawl fewer pages
4. **Poor answers**: Increase chunks or try different embedding
5. **Validation errors**: Check for None values in data

### Error Fixes:

- If getting validation errors with headings, ensure proper None handling
- If OpenRouter errors, check API key and quota
- If embedding errors, check model name and availability

## Customization

### Crawler Settings

- `max_pages`: Maximum pages to crawl (10-30)
- `max_depth`: Link following depth (1-3)

### Chunking Settings

- `chunk_size`: Characters per chunk (400-1500)
- `chunk_overlap`: Overlap between chunks (100-300)

### Search Settings

- `top_k`: Number of chunks to retrieve (3-10)

## Command-Line Usage

### Crawl websites:

```bash
python -m persistent_crawler --url https://docs.example.com --output ./data --max-pages 20 --max-depth 2
```

### Process documents:

```bash
python -m document_processor --crawl-file ./data/example_crawl.json --output-dir ./data --chunk-size 800 --chunk-overlap 150
```

### Create vector DB:

```bash
python -m embedding_manager --chunks-file ./data/processed_chunks.json --persist-dir ./vectordb --model sentence-transformers/all-MiniLM-L6-v2
```

### Query the system:

```bash
python -m rag_query_engine --vectordb-dir ./vectordb --query "How do I install this framework?" --top-k 5
```

## Best Practices

1. Start with smaller crawls (10-20 pages)
2. Use appropriate chunk sizes for your content type:
   - Technical docs: 400-600 characters
   - Narrative docs: 800-1200 characters
3. Increase chunk overlap for complex documents
4. Use OpenRouter API key with sufficient quota
5. Try different embedding models for better results
6. Keep vector databases separate for different documentation sites
