# EmbeddingGemma Powered RAG Assistant
CS 410, Professor Pablo Robles-Granda

Justin Kobza, John Ludeke, Daniel Vlassov, Amber Wilt  
University of Illinois Urbana-Champaign

## Overview

This repository implements a RAG (Retrieval-Augmented Generation) system for scientific document retrieval. The system downloads machine learning papers from arXiv, processes their LaTeX sources, creates text embeddings, and provides semantic search capabilities for finding relevant information across multiple documents.

## Week 1

### What We Built

Week 1 focuses on the data processing pipeline: downloading papers, extracting text, creating searchable chunks, and building a vector database for retrieval.

**Components Implemented:**
- arXiv API client for downloading ML papers
- LaTeX text extraction and cleaning pipeline
- Text chunking strategy with overlap
- TF-IDF embedding generation and transformer-based embeddings (EmbeddingGemma/all-MiniLM-L6-v2)
- FAISS vector store with metadata tracking
- Complete document retrieval pipeline

### How to Run

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Run the Full Pipeline Demo

```bash
python run_full_demo.py
```

**What this does:**
- Searches arXiv for ML papers about "transformer attention"
- Downloads the first paper's LaTeX source
- Extracts and cleans the text content
- Splits the document into overlapping chunks
- Generates TF-IDF embeddings for each chunk
- Builds a FAISS vector store for fast similarity search
- Tests retrieval with sample queries

**Expected Result:**
- Downloads 2 papers from arXiv
- Processes 20-100+ text chunks depending on paper length (uses 200-token chunks with 50-token overlap)
- Creates TF-IDF embeddings (dimension varies by vocabulary size, typically 1000-3000 for full papers)
- Successfully retrieves relevant chunks for test queries
- Shows similarity scores and retrieved text snippets

#### 3. Run Component Tests

```bash
python demo_rag.py
```

This runs individual component tests to verify each piece works correctly.

### File Structure and Purpose

```
src/
├── arxiv_client.py          # Downloads papers from arXiv API
├── latex_parser.py          # Extracts clean text from LaTeX sources
├── text_chunker.py          # Splits documents into overlapping chunks
├── simple_embedding.py      # Generates TF-IDF embeddings
├── embedding_pipeline.py    # Generates transformer-based embeddings (EmbeddingGemma)
├── vector_store.py          # FAISS-based vector database
└── rag_pipeline.py          # Main pipeline that coordinates everything

run_full_demo.py             # Complete end-to-end demonstration
demo_rag.py                  # Component testing and validation
test_rag_system.py           # Comprehensive system tests
requirements.txt             # Python dependencies
```

**Detailed File Descriptions:**

- **`arxiv_client.py`**: Handles arXiv API communication. Searches for papers by category (cs.LG, cs.AI, etc.) and downloads both PDF and LaTeX source files. Includes rate limiting to be respectful to arXiv servers.

- **`latex_parser.py`**: Processes LaTeX documents to extract clean text. Removes LaTeX commands, equations, and formatting while preserving document structure. Identifies sections like Introduction, Method, Results, etc.

- **`text_chunker.py`**: Splits documents into overlapping chunks for embedding generation. Uses sentence-based splitting with configurable chunk size and overlap. Preserves metadata like section titles and document IDs.

- **`simple_embedding.py`**: Creates TF-IDF embeddings from text chunks. This is a lightweight alternative for testing and prototyping. Builds vocabulary from all documents and calculates term frequency-inverse document frequency scores with L2 normalization.

- **`embedding_pipeline.py`**: Generates high-quality embeddings using transformer models (SentenceTransformer). Supports EmbeddingGemma (`google/gemma-2-2b-it`) and falls back to `all-MiniLM-L6-v2` (384-dimensional). Includes batch processing, GPU acceleration, and progress tracking for efficient embedding generation at scale.

- **`vector_store.py`**: Implements FAISS vector database for fast similarity search. Stores embeddings with metadata, handles cosine similarity calculations, and provides top-k retrieval functionality. Supports multiple index types (flat, IVF, HNSW) and includes save/load functionality for persistence.

- **`rag_pipeline.py`**: Orchestrates the entire pipeline from paper download to retrieval. Provides high-level interface for building complete systems and querying the vector store.

### Architecture

```
arXiv Papers → LaTeX Extraction → Text Chunking → Embeddings → Vector Store → Retrieval
```

1. **Download**: Search and download papers from arXiv
2. **Parse**: Extract clean text from LaTeX sources
3. **Chunk**: Split documents into overlapping segments
4. **Embed**: Generate vector representations of text chunks
5. **Store**: Build searchable vector database
6. **Retrieve**: Find relevant chunks for user queries

### Next Steps (Week 2)

- Integrate Qwen-3 LLM for response generation
- Implement proper citation formatting
- Add query preprocessing and expansion
- Build evaluation metrics
- Create simple UI demo