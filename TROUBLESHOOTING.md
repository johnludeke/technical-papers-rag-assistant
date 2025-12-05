# Troubleshooting Guide

## Common Issues and Solutions

### 1. Qwen LLM Not Loading

**Symptoms:**
- Pipeline falls back to embedding-only mode
- Generation fails with dimension mismatch errors

**Possible Causes:**
- EmbeddingGemma produces 384-dim embeddings
- Vector store might have different dimensions from previous runs

**Solution:**
```bash
# Delete old vector store and rebuild
rm -rf data/vector_stores/

# Run demo again to rebuild with correct dimensions
python demo.py
```

**Check embedding dimension:**
The demo now prints embedding dimension on startup. EmbeddingGemma-300m should show `384`.

### 2. Irrelevant Chunks Retrieved

**Recent Fix (Applied):**
- Increased overlap from 50 → 128 tokens
- Keeps chunk size at 512 tokens
- More context overlap improves relevance

**Further Improvements to Try:**

**Option A: Increase chunk size**
Edit `src/text_chunker.py` line 28:
```python
chunk_size: int = 768,  # More context per chunk
overlap_size: int = 192,  # 25% overlap
```

**Option B: Semantic chunking**
Current chunking splits by token count. Consider splitting by:
- Section boundaries (Introduction, Methods, Results)
- Paragraph boundaries
- Sentence boundaries with minimum token threshold

### 3. HuggingFace Authentication

**Error:** `Access denied to EmbeddingGemma`

**Solution:**
1. Go to https://huggingface.co/google/embeddinggemma-300m
2. Accept the model license agreement
3. Generate access token: https://huggingface.co/settings/tokens
4. Run: `huggingface-cli login`
5. Paste your token

### 4. PyTorch/Keras Conflicts

**Error:** `Failed to import transformers.modeling_tf_utils`

**Solution:**
```bash
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge tf-keras
```

### 5. Out of Memory

**Symptoms:**
- System hangs during model loading
- Killed during generation

**Solutions:**

**Use smaller models:**
Edit `demo.py` line 29:
```python
llm_model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller, faster
```

**Enable 8-bit quantization:**
Edit `src/rag_pipeline.py` in `__init__`:
```python
self.llm_generator = LLMGenerator(
    model_name=llm_model_name,
    load_in_8bit=True  # Reduces memory by ~50%
)
```

**Disable LLM entirely:**
```bash
# Just test retrieval
python demo.py
# Will auto-fallback to embedding-only mode
```

### 6. Slow Generation

**Expected Times:**
- CPU: 2-10 seconds per query
- GPU: 0.5-2 seconds per query

**If slower:**

**Reduce output length:**
Edit `demo.py` line 107:
```python
generation_config=GenerationConfig(
    max_new_tokens=200,  # Reduce from 300
    temperature=0.7
)
```

**Use smaller model:**
```python
llm_model_name="Qwen/Qwen2.5-0.5B-Instruct"
```

## Debugging Tips

### Check System Status

**Embedding dimension:**
```python
python -c "from src.embedding_pipeline import EmbeddingPipeline; p = EmbeddingPipeline('google/embeddinggemma-300m'); print(f'Dim: {p.model.get_sentence_embedding_dimension()}')"
```

**Vector store stats:**
```python
from src.vector_store import FAISSVectorStore
store = FAISSVectorStore()
store.load("data/vector_stores/ml_papers")
print(store.get_stats())
```

**Memory usage:**
```bash
# While demo is running
ps aux | grep python
```

### Test Individual Components

**Test embeddings only:**
```bash
python test_components.py
```

**Test retrieval only:**
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_llm=False)
pipeline.load_vector_store("ml_papers")
result = pipeline.query("test query", top_k=3)

for chunk in result.retrieved_chunks:
    print(f"Score: {chunk.score:.4f}")
    print(f"Text: {chunk.text[:100]}...")
    print()
```

## Performance Tuning

### Improve Retrieval Quality

**1. Adjust top_k:**
```python
# Retrieve more chunks for better context
result = pipeline.query("question", top_k=10)
```

**2. Add score threshold:**
```python
# Only use high-quality matches
result = pipeline.query("question", score_threshold=0.7)
```

**3. Rebuild with better chunking:**
```bash
# After editing text_chunker.py
rm -rf data/vector_stores/
python demo.py
```

### Speed Up Generation

**1. Batch queries (if doing multiple):**
Process multiple questions together instead of one-by-one.

**2. Cache results:**
Store generated responses to avoid re-computing.

**3. Use GPU:**
PyTorch will automatically use GPU if available. Check with:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

## Getting Help

If issues persist:

1. Check error messages carefully
2. Verify all requirements installed: `pip list`
3. Check Python version: `python --version` (should be 3.11.7)
4. Look at system resources: `htop` or Activity Monitor
5. Review this guide for similar issues

## Current Known Issues

- **Qwen initialization**: May fail on some systems, falls back to embedding-only mode
- **Chunk relevance**: Improved with 128-token overlap, may need further tuning
- **First run slow**: Downloads models and papers, subsequent runs are faster
