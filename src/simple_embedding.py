"""
Simple embedding pipeline using basic methods for demonstration.
"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import pickle


@dataclass
class SimpleEmbeddingResult:
    """Result of simple embedding generation."""
    chunk_id: str
    embedding: np.ndarray
    metadata: Dict


class SimpleEmbeddingPipeline:
    """Simple embedding pipeline using TF-IDF and basic methods."""
    
    def __init__(self):
        """Initialize the simple embedding pipeline."""
        self.vocab = {}
        self.idf_scores = {}
        self.vocab_size = 0
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in set(words):  # Unique words per document
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Build vocabulary
        self.vocab = {word: idx for idx, word in enumerate(sorted(word_counts.keys()))}
        self.vocab_size = len(self.vocab)
        
        # Calculate IDF scores
        total_docs = len(texts)
        for word in self.vocab:
            doc_count = sum(1 for text in texts if word in text.lower().split())
            self.idf_scores[word] = np.log(total_docs / (doc_count + 1))
    
    def _text_to_tfidf(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        words = text.lower().split()
        word_counts = {}
        
        # Count words
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create TF-IDF vector
        tfidf_vector = np.zeros(self.vocab_size)
        
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = count / len(words)  # Term frequency
                idf = self.idf_scores[word]  # Inverse document frequency
                tfidf_vector[idx] = tf * idf
        
        # Normalize vector
        norm = np.linalg.norm(tfidf_vector)
        if norm > 0:
            tfidf_vector = tfidf_vector / norm
        
        return tfidf_vector
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts."""
        # Build vocabulary first
        self._build_vocab(texts)
        
        # Generate embeddings
        embeddings = []
        for text in texts:
            embedding = self._text_to_tfidf(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self._text_to_tfidf(text)
    
    def process_chunks(self, chunks) -> List[SimpleEmbeddingResult]:
        """Process text chunks and generate embeddings."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            result = SimpleEmbeddingResult(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                metadata={
                    'arxiv_id': chunk.arxiv_id,
                    'section_title': chunk.section_title,
                    'section_type': chunk.section_type,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'token_count': chunk.token_count
                }
            )
            results.append(result)
        
        return results


def main():
    """Test the simple embedding pipeline."""
    pipeline = SimpleEmbeddingPipeline()
    
    # Test texts
    texts = [
        "This is a sample text about machine learning and neural networks.",
        "Another sample text discussing transformers and attention mechanisms.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons."
    ]
    
    # Generate embeddings
    embeddings = pipeline.generate_embeddings(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings[0].shape}")
    
    # Test similarity
    query = "machine learning neural networks"
    query_embedding = pipeline.generate_embedding(query)
    
    print(f"\nQuery: {query}")
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(embeddings):
        similarity = np.dot(embedding, query_embedding)
        similarities.append((i, similarity, texts[i]))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nSimilarities:")
    for i, (idx, sim, text) in enumerate(similarities):
        print(f"{i+1}. Similarity: {sim:.4f} - {text}")


if __name__ == "__main__":
    main()
