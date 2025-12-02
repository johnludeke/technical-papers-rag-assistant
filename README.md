# EmbeddingGemma Powered RAG Assistant

A Retrieval-Augmented Generation (RAG) system for scientific document search and question answering with proper source citations.

**CS 410 Final Project** | University of Illinois Urbana-Champaign
Justin Kobza, John Ludeke, Daniel Vlassov, Amber Wilt

---

## What This Does

Ask questions about machine learning papers and get answers with citations:

```
Q: "What is the attention mechanism?"

A: The attention mechanism [1] is a technique that allows neural networks
   to focus on different parts of the input when producing output. In
   transformers, self-attention [1] enables the model to weigh different
   words. Multi-head attention [2] extends this by learning multiple
   patterns in parallel.

Sources:
[1] Attention Is All You Need (arxiv:1706.03762)
    Section: Introduction | Relevance: 0.92
[2] Attention Is All You Need (arxiv:1706.03762)
    Section: Model Architecture | Relevance: 0.88
```

---

## Quick Start

### 1. Install

```bash
# Built on Python 3.11.7
pip install -r requirements.txt

# You will need to go to HugginFace and generate an access token for your account that has access granted to EmbeddingGemma, then paste it here.
hf auth logins

```

### 2. Test (No Downloads)

```bash
python test_components.py
```

Expected: `✅ 4/4 tests passed`

### 3. Run Demo (Downloads Models)

```bash
python demo.py
```

First run downloads ~3-6 GB of models and papers (10-30 minutes).

---

## How It Works

```
User Question
    ↓
Convert to embedding vector
    ↓
Search vector database (FAISS)
    ↓
Retrieve relevant paper chunks
    ↓
Format with citation markers [1], [2]...
    ↓
Send to LLM (Qwen-3)
    ↓
Generate answer with inline citations
    ↓
Map citations to source papers
```

---

## Project Structure

```
├── src/                      # Core library
│   ├── arxiv_client.py      # Download papers
│   ├── latex_parser.py      # Extract text from LaTeX
│   ├── text_chunker.py      # Split into chunks
│   ├── embedding_pipeline.py # Create embeddings
│   ├── vector_store.py      # FAISS database
│   ├── llm_generator.py     # LLM + citations
│   ├── evaluation.py        # Metrics
│   └── rag_pipeline.py      # Orchestration
│
├── demo.py                   # Main demo
├── test_components.py        # Quick tests
├── verify_week2_complete.py  # Full verification
│
├── docs/                     # Technical docs
│   ├── ARCHITECTURE.md       # System design
│   ├── WEEK_2_SUMMARY.md     # Implementation details
│   └── FILES.md              # File reference
│
└── archive/                  # Old deliverables
    └── week1_deliverables/   # Week 1 reports
```

---

## Usage

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()
pipeline.load_vector_store("ml_papers")

# Ask question
response = pipeline.query_with_generation(
    "How does multi-head attention work?",
    top_k=5
)

print(response.response)
```

### Retrieval Only (Faster, No LLM)

```python
pipeline = RAGPipeline(use_llm=False)
pipeline.load_vector_store("ml_papers")

result = pipeline.query("What is attention?", top_k=5)
print(result.context)
```

### Build Your Own Database

```python
pipeline = RAGPipeline()
pipeline.build_complete_pipeline(
    query="neural networks deep learning",
    max_papers=10,
    store_name="my_papers"
)
```

---

## Key Features

✅ **Document Processing**
- Downloads papers from arXiv
- Extracts text from LaTeX
- Splits into 200-token chunks with overlap

✅ **Semantic Search**
- 384-dimensional embeddings
- FAISS vector database
- Cosine similarity search

✅ **LLM Generation**
- Qwen-3 (1.5B parameters)
- Inline citations [1], [2], [3]...
- Source attribution

✅ **Evaluation**
- Precision@K, Recall@K, MRR, NDCG
- Latency tracking
- Citation accuracy

---

## Performance

**Typical Latency:**
- CPU: 2-10 seconds per query
- GPU: 0.5-2 seconds per query

**Memory:**
- Embedding model: ~200 MB
- LLM (1.5B): ~3-6 GB (FP16) or ~1.5-3 GB (8-bit)
- Vector store: ~100 MB per 1000 papers

---

## Troubleshooting

### Out of Memory?

Use smaller model:
```python
pipeline = RAGPipeline(
    llm_model_name="Qwen/Qwen2.5-0.5B-Instruct"
)
```

Or disable LLM:
```python
pipeline = RAGPipeline(use_llm=False)
```

### Slow Generation?

1. Use GPU (automatic if available)
2. Reduce output length: `GenerationConfig(max_new_tokens=200)`
3. Use smaller model (0.5B instead of 1.5B)

### No Papers?

First run auto-downloads. To manually build:
```python
pipeline.build_complete_pipeline(query="your topic", max_papers=5)
```

---

## Configuration

### Change Models

```python
pipeline = RAGPipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"
)
```

### Tune Generation

```python
from src.llm_generator import GenerationConfig

config = GenerationConfig(
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9
)

response = pipeline.query_with_generation(
    "Question?",
    generation_config=config
)
```

---

## Testing

**Quick test (no downloads):**
```bash
python test_components.py
```

**Full verification:**
```bash
python verify_week2_complete.py
```

**Run demo:**
```bash
python demo.py
```

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design decisions
- **[docs/WEEK_2_SUMMARY.md](docs/WEEK_2_SUMMARY.md)** - Detailed implementation summary
- **[docs/FILES.md](docs/FILES.md)** - Complete file reference

---

## Development Status

### ✅ Completed (Weeks 1-2)
- arXiv downloading and LaTeX parsing
- Text chunking and embeddings
- Vector database (FAISS)
- LLM integration (Qwen-3)
- Citation tracking
- Evaluation metrics
- End-to-end pipeline

### 🚧 Week 3 (In Progress)
- Model quantization
- Inference optimization
- Fine-tuning
- Advanced prompting

### 📅 Week 4 (Planned)
- Web UI
- Interactive citations
- Export functionality
- User studies

---

## Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Test components (fast)
python test_components.py

# Run full demo (slow, downloads models)
python demo.py

# Verify everything works
python verify_week2_complete.py
```

---

## Team

- **Justin Kobza** - jkobza2@illinois.edu
- **John Ludeke** - jludeke2@illinois.edu
- **Daniel Vlassov** - dvlas2@illinois.edu
- **Amber Wilt** - anwilt2@illinois.edu

**Instructor:** Professor Pablo Robles-Granda
**Course:** CS 410, Fall 2024
**University:** University of Illinois Urbana-Champaign

---

## Citation

```bibtex
@misc{kobza2024embeddinggemma,
  title={EmbeddingGemma Powered RAG Assistant},
  author={Justin Kobza and John Ludeke and Daniel Vlassov and Amber Wilt},
  year={2024},
  institution={University of Illinois Urbana-Champaign}
}
```

---

For questions, see documentation in `docs/` or contact the team.
