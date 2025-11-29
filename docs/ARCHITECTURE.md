# System Architecture - Week 2

## Overview

The EmbeddingGemma Powered RAG Assistant is now a complete retrieval-augmented generation system with citation support.

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                      (Query Input)                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                                │
│  (rag_pipeline.py)                                              │
│                                                                  │
│  • query()                    ← Retrieval only                  │
│  • query_with_generation()    ← Full RAG with LLM              │
└───┬──────────────────────────────────────────────────────────┬───┘
    │                                                          │
    │ 1. Query Embedding                                       │
    ▼                                                          │
┌──────────────────────────────────┐                          │
│   EMBEDDING PIPELINE             │                          │
│   (embedding_pipeline.py)        │                          │
│                                  │                          │
│   • SentenceTransformer          │                          │
│   • EmbeddingGemma               │                          │
│   • 384-dim vectors              │                          │
└────────────┬─────────────────────┘                          │
             │                                                 │
             │ 2. Vector Search                               │
             ▼                                                 │
┌──────────────────────────────────┐                          │
│   VECTOR STORE (FAISS)           │                          │
│   (vector_store.py)              │                          │
│                                  │                          │
│   • Cosine similarity search     │                          │
│   • Top-K retrieval              │                          │
│   • Metadata tracking            │                          │
└────────────┬─────────────────────┘                          │
             │                                                 │
             │ 3. Retrieved Chunks + Metadata                 │
             ▼                                                 │
┌──────────────────────────────────────────────────────────────┐  │
│   CONTEXT ASSEMBLY                                           │  │
│   (llm_generator.py)                                        │  │
│                                                              │  │
│   • Format chunks with citations [1], [2], [3]...          │  │
│   • Add metadata (paper ID, section, score)                 │  │
│   • Build structured context                                │  │
└────────────┬─────────────────────────────────────────────────┘  │
             │                                                     │
             │ 4. Formatted Context                               │
             ▼                                                     │
┌──────────────────────────────────────────────────────────────┐  │
│   PROMPT CONSTRUCTION                                        │  │
│   (llm_generator.py)                                        │  │
│                                                              │  │
│   System Prompt:                                            │  │
│   "You are a research assistant. Answer based on context    │  │
│    and cite sources using [1], [2] markers..."              │  │
│                                                              │  │
│   User Prompt:                                              │  │
│   "Context: [1] (Source: paper1) Text here...               │  │
│             [2] (Source: paper2) More text...               │  │
│    Question: {user_query}"                                  │  │
└────────────┬─────────────────────────────────────────────────┘  │
             │                                                     │
             │ 5. Complete Prompt                                 │
             ▼                                                     │
┌──────────────────────────────────────────────────────────────┐  │
│   LLM GENERATION                                             │  │
│   (llm_generator.py + Qwen-3)                               │  │
│                                                              │  │
│   • Load Qwen-3 model (1.5B params)                         │  │
│   • Generate with temperature=0.7                           │  │
│   • Use citation markers in response                        │  │
│   • Extract generated text                                  │  │
└────────────┬─────────────────────────────────────────────────┘  │
             │                                                     │
             │ 6. Generated Response with [1], [2] citations      │
             ▼                                                     │
┌──────────────────────────────────────────────────────────────┐  │
│   CITATION MAPPING                                           │  │
│   (llm_generator.py)                                        │  │
│                                                              │  │
│   • Map [1] → Paper A, Section 2.1, Score: 0.92            │  │
│   • Map [2] → Paper B, Introduction, Score: 0.87           │  │
│   • Format source attribution                               │  │
└────────────┬─────────────────────────────────────────────────┘  │
             │                                                     │
             │ 7. Response + Full Citation Details                │
             ▼                                                     │
┌──────────────────────────────────────────────────────────────┐  │
│   RESPONSE FORMATTING                                        │  │
│   (llm_generator.py)                                        │  │
│                                                              │  │
│   Response: "The attention mechanism [1] allows..."         │  │
│                                                              │  │
│   Sources:                                                   │  │
│   [1] Attention Is All You Need (arxiv:1706.03762)         │  │
│       Section: Introduction, Relevance: 0.92                │  │
└────────────┬─────────────────────────────────────────────────┘  │
             │                                                     │
             ▼                                                     │
┌──────────────────────────────────────────────────────────────┐  │
│   EVALUATION (Optional)                                      │◄─┘
│   (evaluation.py)                                           │
│                                                              │
│   • Precision@K, Recall@K                                   │
│   • MRR, NDCG@K                                            │
│   • Latency measurement                                     │
│   • Citation accuracy                                       │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                              │
│                                                              │
│  • Natural language answer                                  │
│  • Inline citations                                         │
│  • Source attribution                                       │
│  • Performance metrics                                      │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow: Week 1 Components

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   arXiv     │────▶│   LaTeX      │────▶│    Text     │
│   Client    │     │   Parser     │     │   Chunker   │
│             │     │              │     │             │
│ Download    │     │ Extract text │     │ Split into  │
│ papers      │     │ Clean LaTeX  │     │ chunks      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │  Embedding  │
                                          │  Pipeline   │
                                          │             │
                                          │ Generate    │
                                          │ vectors     │
                                          └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Vector    │
                                          │   Store     │
                                          │   (FAISS)   │
                                          └─────────────┘
```

## Component Responsibilities

### Week 1 (Data Pipeline)

| Component | File | Responsibility |
|-----------|------|----------------|
| arXiv Client | `arxiv_client.py` | Download papers from arXiv |
| LaTeX Parser | `latex_parser.py` | Extract clean text from LaTeX |
| Text Chunker | `text_chunker.py` | Split into overlapping chunks |
| Embedding Pipeline | `embedding_pipeline.py` | Generate embeddings |
| Vector Store | `vector_store.py` | Store and search vectors |

### Week 2 (Generation & Evaluation)

| Component | File | Responsibility |
|-----------|------|----------------|
| LLM Generator | `llm_generator.py` | Generate responses with citations |
| Evaluation | `evaluation.py` | Measure system performance |
| RAG Pipeline | `rag_pipeline.py` | Orchestrate end-to-end flow |

## Key Design Decisions

### 1. Citation System

**Design:** Inline citation markers [1], [2], [3]...

**Rationale:**
- Easy for LLM to learn and use
- Familiar format for academic users
- Simple to parse and validate
- Maps naturally to source documents

### 2. Two-Stage Architecture

**Design:** Separate retrieval and generation stages

**Rationale:**
- Can use different models for each stage
- Easier to optimize independently
- Can test retrieval without LLM
- Supports embedding-only mode

### 3. Metadata Tracking

**Design:** Store paper ID, section, score with each chunk

**Rationale:**
- Enables source attribution
- Supports evaluation
- Helps users verify information
- Allows for re-ranking

### 4. Evaluation Framework

**Design:** Multiple metrics (Precision, Recall, MRR, NDCG, Latency)

**Rationale:**
- Comprehensive quality assessment
- Industry-standard metrics
- Supports A/B testing
- Tracks performance over time

## Performance Characteristics

### Latency Breakdown (Typical Query)

```
Total Time: ~2-5 seconds (CPU) / ~0.5-1 second (GPU)

┌───────────────────────────────────────────────────┐
│ Query Embedding       │ ███░░░░░░░░░░░░   50ms   │
├───────────────────────────────────────────────────┤
│ Vector Search         │ ████░░░░░░░░░░░   80ms   │
├───────────────────────────────────────────────────┤
│ Context Assembly      │ █░░░░░░░░░░░░░░   10ms   │
├───────────────────────────────────────────────────┤
│ LLM Generation (CPU)  │ ████████████  2000ms     │
│ LLM Generation (GPU)  │ ████░░░░░░░░  400ms      │
├───────────────────────────────────────────────────┤
│ Citation Mapping      │ █░░░░░░░░░░░░░   10ms   │
└───────────────────────────────────────────────────┘

Note: LLM generation dominates latency
```

### Memory Usage

```
Embedding Model (MiniLM):    ~200 MB
LLM (Qwen-3 1.5B FP16):      ~3 GB
LLM (Qwen-3 1.5B 8-bit):     ~1.5 GB
Vector Store (1000 papers):   ~100 MB
──────────────────────────────────────
Total (FP16):                 ~3.3 GB
Total (8-bit):                ~1.8 GB
```

## Scalability Considerations

### Current Limitations

- **LLM Generation:** Bottleneck on CPU (2-10 seconds/query)
- **Vector Store:** In-memory, limited by RAM
- **Batch Processing:** Processes one query at a time

### Week 3 Optimization Targets

1. **Model Quantization:** 8-bit or 4-bit to reduce memory
2. **KV Cache:** Faster generation for similar queries
3. **Batch Processing:** Multiple queries in parallel
4. **GPU Utilization:** Better use of available hardware
5. **Context Compression:** Reduce tokens sent to LLM

## Example Flow: "What is attention mechanism?"

```
1. User Query
   Input: "What is attention mechanism?"

2. Query Embedding
   Vector: [0.123, -0.456, 0.789, ...] (384-dim)

3. Vector Search
   Top 5 chunks retrieved:
   - Chunk A: Score 0.92 (Introduction, Paper 1706.03762)
   - Chunk B: Score 0.87 (Architecture, Paper 1706.03762)
   - Chunk C: Score 0.81 (Related Work, Paper 1810.04805)
   - Chunk D: Score 0.79 (Experiments, Paper 1706.03762)
   - Chunk E: Score 0.75 (Conclusion, Paper 1706.03762)

4. Context Assembly
   [1] (Source: 1706.03762, Section: Introduction, Score: 0.92)
   The attention mechanism allows models to focus on...

   [2] (Source: 1706.03762, Section: Architecture, Score: 0.87)
   Multi-head attention consists of several layers...

   [continues for all 5 chunks]

5. Prompt Construction
   System: You are a research assistant...
   Context: [formatted chunks above]
   Question: What is attention mechanism?

6. LLM Generation
   "The attention mechanism [1] is a neural network component that
   allows models to selectively focus on different parts of the input
   when producing output. In transformers, self-attention [1] enables
   the model to weigh the importance of different words in a sequence.
   Multi-head attention [2] extends this by learning multiple attention
   patterns in parallel, allowing the model to attend to information
   from different representation subspaces [2]."

7. Citation Mapping
   [1] → Paper: Attention Is All You Need (1706.03762)
         Section: Introduction
         Relevance: 0.92

   [2] → Paper: Attention Is All You Need (1706.03762)
         Section: Architecture
         Relevance: 0.87

8. Final Output
   [Formatted response with sources displayed]
```

## Technology Stack

### Core Dependencies

```
Python 3.8+
├── PyTorch 2.1.0              (Deep learning framework)
├── Transformers 4.35.2         (Hugging Face models)
├── Sentence-Transformers 2.2.2 (Embeddings)
├── FAISS 1.7.4                 (Vector search)
├── Accelerate 0.24.1           (Model optimization)
└── NumPy, Pandas, tqdm         (Utilities)
```

### Models

```
Embeddings:
├── sentence-transformers/all-MiniLM-L6-v2 (384-dim, 80MB)
└── google/gemma-2-2b-it (alternative, larger)

LLM:
├── Qwen/Qwen2.5-1.5B-Instruct (default, 1.5B params)
├── Qwen/Qwen2.5-0.5B-Instruct (smaller, faster)
└── Any compatible HuggingFace causal LM
```

## Future Enhancements (Week 3+)

### Planned

1. **Model Optimization**
   - Quantization (8-bit, 4-bit)
   - Flash attention
   - Better batching

2. **Retrieval Improvements**
   - Re-ranking of results
   - Query expansion
   - Hybrid search (dense + sparse)

3. **User Interface**
   - Web interface
   - Interactive citations
   - Query history

4. **Evaluation**
   - More comprehensive benchmarks
   - User studies
   - A/B testing framework

### Possible

- Fine-tuning on domain data
- Multi-document reasoning
- Conversation history
- Document upload interface
- Export functionality (PDF, Markdown)

## Conclusion

Week 2 delivers a complete, working RAG system with:
- End-to-end query → response flow
- Proper citation tracking
- Comprehensive evaluation
- Modular, extensible architecture

The system is ready for optimization (Week 3) and deployment (Week 4).
