# Project File Manifest

Complete listing of all project files with descriptions.

## Week 2 New Files

### Core Implementation

| File | Size | Description |
|------|------|-------------|
| `src/llm_generator.py` | 11.6 KB | LLM integration with Qwen-3, citation formatting, and response generation |
| `src/evaluation.py` | 13.9 KB | Evaluation framework with Precision@K, Recall@K, MRR, NDCG, latency, and citation metrics |

### Demo & Testing

| File | Size | Description |
|------|------|-------------|
| `week_2_demo.py` | 6.2 KB | Complete Week 2 demonstration with retrieval, generation, and evaluation |
| `test_week2_components.py` | 5.7 KB | Component-level tests for Week 2 features |
| `verify_week2_complete.py` | 12.5 KB | Comprehensive verification script for Week 2 objectives |

### Documentation

| File | Size | Description |
|------|------|-------------|
| `WEEK_2_SUMMARY.md` | 9.8 KB | Technical summary of Week 2 implementation |
| `QUICKSTART_WEEK2.md` | 6.5 KB | Quick start guide for running Week 2 demo |
| `ARCHITECTURE.md` | 13.2 KB | System architecture diagrams and design decisions |
| `FILES.md` | This file | Complete file manifest |

## Week 2 Updated Files

| File | Changes | Description |
|------|---------|-------------|
| `src/rag_pipeline.py` | Added LLM integration | Added `query_with_generation()`, LLM initialization, generation config |
| `requirements.txt` | Added dependencies | Added accelerate, bitsandbytes for model optimization |
| `README.md` | Added Week 2 section | Documented Week 2 features, usage, and architecture |

## Week 1 Files (Unchanged)

### Core Components

| File | Size | Description |
|------|------|-------------|
| `src/__init__.py` | 31 B | Package initialization |
| `src/arxiv_client.py` | 6.2 KB | arXiv API client for downloading papers |
| `src/latex_parser.py` | 10.3 KB | LaTeX text extraction and cleaning |
| `src/text_chunker.py` | 9.9 KB | Text chunking with overlap strategy |
| `src/simple_embedding.py` | 5.0 KB | TF-IDF embedding generation |
| `src/embedding_pipeline.py` | 10.2 KB | Transformer-based embedding generation |
| `src/vector_store.py` | 11.6 KB | FAISS vector store for similarity search |

### Week 1 Demos

| File | Size | Description |
|------|------|-------------|
| `run_full_demo.py` | 5.4 KB | Week 1 full pipeline demonstration |
| `demo_rag.py` | 8.0 KB | Week 1 component testing |
| `test_rag_system.py` | 3.9 KB | Week 1 system tests |
| `simple_test.py` | 4.1 KB | Simple smoke tests |

### Week 1 Documentation

| File | Size | Description |
|------|------|-------------|
| `week_1.tex` | 4.4 KB | Week 1 progress report (LaTeX source) |
| `week_1.pdf` | 313 KB | Week 1 progress report (compiled) |

## Project Configuration

| File | Description |
|------|-------------|
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore patterns |
| `.git/` | Git repository data |

## Data Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Downloaded papers and artifacts |
| `data/papers/` | arXiv papers (PDF and LaTeX) |
| `data/vector_stores/` | Saved FAISS vector stores |

## File Dependencies

### Import Chain

```
week_2_demo.py
    └── src/rag_pipeline.py
        ├── src/arxiv_client.py
        ├── src/latex_parser.py
        ├── src/text_chunker.py
        ├── src/embedding_pipeline.py
        ├── src/vector_store.py
        └── src/llm_generator.py ← NEW
            └── transformers (HuggingFace)

src/evaluation.py ← NEW
    └── numpy
```

## File Sizes Summary

```
Total Project Size: ~600 KB (excluding data and models)

Week 1 Components:    ~70 KB
Week 2 Components:    ~50 KB
Documentation:        ~40 KB
Demo Scripts:         ~30 KB
Configuration:        ~5 KB
```

## Usage by Task

### Quick Test (No Downloads)
```
test_week2_components.py
verify_week2_complete.py
```

### Full Demo (Downloads Models)
```
week_2_demo.py
```

### Week 1 Testing
```
run_full_demo.py
demo_rag.py
test_rag_system.py
```

### Development
```
src/llm_generator.py      ← LLM integration
src/evaluation.py         ← Metrics
src/rag_pipeline.py       ← Main pipeline
```

### Documentation
```
README.md                 ← Main documentation
WEEK_2_SUMMARY.md         ← Technical details
QUICKSTART_WEEK2.md       ← Getting started
ARCHITECTURE.md           ← System design
FILES.md                  ← This file
```

## Code Statistics

### Lines of Code (excluding comments)

| Component | LOC | Files |
|-----------|-----|-------|
| Week 1 Core | ~1,500 | 7 files |
| Week 2 Core | ~800 | 2 files |
| Demos & Tests | ~600 | 6 files |
| **Total** | **~2,900** | **15 files** |

### Function Count

| Component | Functions | Classes |
|-----------|-----------|---------|
| Week 1 Core | ~40 | ~8 |
| Week 2 Core | ~20 | ~4 |
| **Total** | **~60** | **~12** |

## File Status

### Stable (No Changes Expected)
- All Week 1 core files
- Week 1 documentation

### Active Development (Week 2)
- `src/llm_generator.py`
- `src/evaluation.py`
- `src/rag_pipeline.py`

### Future Updates (Week 3)
- Optimization scripts
- Fine-tuning code
- UI components
- Week 3 documentation

## File Relationships

### Dataclass Definitions

```
src/arxiv_client.py:
  - PaperInfo

src/latex_parser.py:
  - ProcessedDocument

src/text_chunker.py:
  - TextChunk

src/embedding_pipeline.py:
  - EmbeddingResult

src/vector_store.py:
  - VectorStoreResult

src/rag_pipeline.py:
  - RAGResult

src/llm_generator.py:
  - GenerationConfig
  - GeneratedResponse

src/evaluation.py:
  - EvaluationMetrics
  - TestQuery
```

### Main Classes

```
ArxivClient          ← Downloads papers
LatexParser          ← Extracts text
TextChunker          ← Splits text
EmbeddingPipeline    ← Generates embeddings
FAISSVectorStore     ← Stores vectors
RAGPipeline          ← Orchestrates everything
LLMGenerator         ← Generates responses (NEW)
RAGEvaluator         ← Evaluates performance (NEW)
```

## Development Workflow

### Adding New Features

1. **Core Logic**: Add to appropriate `src/*.py` file
2. **Tests**: Add to `test_week2_components.py`
3. **Demo**: Update `week_2_demo.py` if needed
4. **Docs**: Update `README.md` and relevant docs

### Running Tests

```bash
# Quick component tests
python test_week2_components.py

# Full verification
python verify_week2_complete.py

# Full demo (slow, downloads models)
python week_2_demo.py
```

### Modifying Components

```bash
# LLM generation
vim src/llm_generator.py

# Evaluation metrics
vim src/evaluation.py

# Pipeline integration
vim src/rag_pipeline.py
```

## Backup & Version Control

### Important Files to Backup

- All `src/*.py` files
- All `*.md` documentation
- `requirements.txt`
- `week_2_demo.py`
- All test files

### Git Status

```bash
# New files (Week 2)
?? src/llm_generator.py
?? src/evaluation.py
?? week_2_demo.py
?? test_week2_components.py
?? verify_week2_complete.py
?? WEEK_2_SUMMARY.md
?? QUICKSTART_WEEK2.md
?? ARCHITECTURE.md
?? FILES.md

# Modified files
M  src/rag_pipeline.py
M  requirements.txt
M  README.md
```

## File Checksums (for Verification)

To verify file integrity:

```bash
# Week 2 core files should exist
ls -l src/llm_generator.py src/evaluation.py

# Week 2 docs should exist
ls -l WEEK_2_SUMMARY.md QUICKSTART_WEEK2.md ARCHITECTURE.md

# Week 2 tests should exist
ls -l test_week2_components.py verify_week2_complete.py
```

## Summary

**Week 2 Added:**
- 5 new core/test files (~32 KB code)
- 4 documentation files (~30 KB docs)
- Updated 3 existing files
- Total: ~60 KB new content

**Project Now Contains:**
- 15 Python source files
- 8 documentation files
- 4 demo/test scripts
- Complete Week 1 + Week 2 implementation

**Status:** ✅ Week 2 Complete and Verified
