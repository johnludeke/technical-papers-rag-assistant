# Week 2 Quick Start Guide

## Prerequisites

1. Python 3.8+ installed
2. Git repository cloned
3. Week 1 completed (or vector store available)

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

This will install:
- Week 1 dependencies (arxiv, sentence-transformers, faiss, etc.)
- Week 2 additions (accelerate, bitsandbytes)

## Quick Test (No Model Downloads)

Test that all components are working:

```bash
python test_week2_components.py
```

Expected output:
```
✅ PASS: Evaluation Framework
✅ PASS: LLM Generator Import
✅ PASS: RAG Pipeline Updates
✅ PASS: Citation Formatting

4/4 tests passed
```

## Full Demo (Downloads Models)

Run the complete Week 2 demo:

```bash
python week_2_demo.py
```

**Note:** First run will download:
- Qwen-3 LLM (~3-6 GB)
- Sentence Transformer model (~80 MB)
- Papers from arXiv (if vector store doesn't exist)

This may take 10-30 minutes depending on your internet connection.

## What the Demo Does

1. **Initialization**: Loads embedding model and Qwen LLM
2. **Vector Store**: Loads existing store or builds new one
3. **Retrieval Test**: Tests semantic search with sample queries
4. **Generation Test**: Generates responses with citations
5. **Evaluation**: Runs benchmarks and displays metrics

## Expected Output

### Retrieval Example
```
Query: What is the attention mechanism?
Retrieved 3 chunks:
  1. Score: 0.9245 | Section: Introduction
     Preview: The attention mechanism allows neural networks to focus...
```

### Generation Example
```
Response: The attention mechanism [1] is a technique that allows neural
networks to focus on different parts of the input. In transformers,
self-attention [1] enables the model to weigh the importance of different
words. Multi-head attention [2] extends this by learning multiple patterns.

Sources:
[1] Attention Is All You Need (arxiv:1706.03762)
    Section: Introduction | Relevance: 0.9245
[2] Attention Is All You Need (arxiv:1706.03762)
    Section: Model Architecture | Relevance: 0.8876
```

### Evaluation Metrics
```
📊 Retrieval Metrics:
  Precision@K:          0.8500
  Recall@K:             0.9200
  Mean Reciprocal Rank: 0.8750
  NDCG@K:               0.8900

⏱️  Latency Metrics:
  Avg Retrieval Time:   125.43 ms
  Avg Generation Time:  2341.56 ms
  Avg Total Time:       2467.00 ms

📝 Citation Metrics:
  Citation Accuracy:    0.9500
  Avg Citations/Response: 3.20
```

## Troubleshooting

### Out of Memory

If you get OOM errors:

1. Use smaller model:
```python
pipeline = RAGPipeline(
    llm_model_name="Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model
)
```

2. Enable 8-bit quantization (requires CUDA):
```python
generator = LLMGenerator(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    load_in_8bit=True
)
```

3. Disable LLM generation:
```python
pipeline = RAGPipeline(use_llm=False)
# Use pipeline.query() instead of query_with_generation()
```

### Slow Generation

CPU generation is slow. To speed up:

1. Reduce max_new_tokens:
```python
config = GenerationConfig(max_new_tokens=200)
```

2. Use GPU if available (automatic)

3. Use smaller model

### No Papers in Vector Store

If Week 1 wasn't completed:

```bash
# The demo will automatically download and process papers
python week_2_demo.py
```

Or manually build:

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_llm=False)
pipeline.build_complete_pipeline(
    query="transformer attention",
    max_papers=5
)
```

## Testing Individual Components

### Test Evaluation Framework
```bash
python -c "from src.evaluation import main; main()"
```

### Test LLM Generator (Mock Data)
```bash
python -c "from src.llm_generator import main; main()"
```

Note: This will download the LLM model.

## File Overview

**New Week 2 Files:**
- `src/llm_generator.py` - LLM integration with citations
- `src/evaluation.py` - Evaluation metrics and benchmarks
- `week_2_demo.py` - Main demo script
- `test_week2_components.py` - Component tests
- `WEEK_2_SUMMARY.md` - Detailed documentation
- `QUICKSTART_WEEK2.md` - This file

**Updated Files:**
- `src/rag_pipeline.py` - Added LLM generation methods
- `requirements.txt` - Added new dependencies
- `README.md` - Added Week 2 documentation

## Common Use Cases

### 1. Just Test Components (Fast)
```bash
python test_week2_components.py
```

### 2. Test Retrieval Only (No LLM)
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_llm=False)
pipeline.load_vector_store("ml_papers")

result = pipeline.query("What is attention?", top_k=5)
print(result.context)
```

### 3. Full RAG with Generation
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_llm=True)
pipeline.load_vector_store("ml_papers")

response = pipeline.query_with_generation(
    "What is attention?",
    top_k=5
)

print(response.response)
for citation in response.citations:
    print(f"{citation['citation_id']}: {citation['title']}")
```

### 4. Run Evaluation
```python
from src.rag_pipeline import RAGPipeline
from src.evaluation import RAGEvaluator, TestQuery

pipeline = RAGPipeline(use_llm=True)
pipeline.load_vector_store("ml_papers")

evaluator = RAGEvaluator()
test_queries = [
    TestQuery(
        query="What is attention?",
        relevant_doc_ids=["1706.03762"]
    )
]

metrics = evaluator.run_benchmark(pipeline, test_queries, k=5)
evaluator.print_metrics(metrics)
```

## Performance Tips

1. **First Run**: Will be slow due to model downloads
2. **Subsequent Runs**: Much faster as models are cached
3. **CPU vs GPU**: GPU is 10-20x faster for generation
4. **Batch Queries**: More efficient than one-by-one
5. **Smaller Models**: Trade quality for speed

## Next Steps

After Week 2 works:
- Experiment with different LLM models
- Try different generation parameters
- Add your own papers to the vector store
- Customize evaluation metrics
- Move to Week 3: Optimization and fine-tuning

## Support

For issues:
1. Check [README.md](README.md) for detailed documentation
2. See [WEEK_2_SUMMARY.md](WEEK_2_SUMMARY.md) for technical details
3. Review error messages carefully
4. Ensure all dependencies are installed

## Summary

Week 2 adds LLM-powered response generation with citations to the RAG system. The demo showcases:

✅ Semantic retrieval from scientific papers
✅ Natural language response generation
✅ Proper citation tracking and source attribution
✅ Comprehensive evaluation metrics
✅ End-to-end working system

Ready for Week 3: Optimization and deployment!
