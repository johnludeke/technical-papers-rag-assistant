"""
EmbeddingGemma integration and embedding pipeline.
"""
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: np.ndarray
    metadata: Dict


class EmbeddingPipeline:
    """Pipeline for generating embeddings using EmbeddingGemma."""
    
    def __init__(self, 
                 model_name: str = "google/gemma-2-2b-it",
                 device: str = "auto",
                 batch_size: int = 32):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: Name of the EmbeddingGemma model
            device: Device to use for inference ('auto', 'cpu', 'cuda')
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Initialize the model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the EmbeddingGemma model."""
        try:
            print(f"Loading EmbeddingGemma model: {self.model_name}")
            
            # Use SentenceTransformer with EmbeddingGemma
            # Note: We'll use a compatible model for now since EmbeddingGemma might not be directly available
            # You may need to adjust this based on the actual model availability
            
            # Try to load EmbeddingGemma, fallback to a similar model if not available
            try:
                self.model = SentenceTransformer(self.model_name)
            except:
                # Fallback to a good embedding model for scientific text
                print("EmbeddingGemma not available, using alternative model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Move to device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = self.model.to(self.device)
            print(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, 
                                         batch_size=self.batch_size,
                                         show_progress_bar=True,
                                         convert_to_tensor=False)
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise
    
    def process_chunks(self, chunks: List) -> List[EmbeddingResult]:
        """
        Process text chunks and generate embeddings.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of EmbeddingResult objects
        """
        print(f"Processing {len(chunks)} chunks...")
        
        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(texts)
        
        # Create results
        results = []
        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
            result = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata={
                    'arxiv_id': chunks[i].arxiv_id,
                    'section_title': chunks[i].section_title,
                    'section_type': chunks[i].section_type,
                    'chunk_index': chunks[i].chunk_index,
                    'total_chunks': chunks[i].total_chunks,
                    'token_count': chunks[i].token_count
                }
            )
            results.append(result)
        
        return results
    
    def save_embeddings(self, results: List[EmbeddingResult], output_path: str):
        """
        Save embeddings to disk.
        
        Args:
            results: List of EmbeddingResult objects
            output_path: Path to save the embeddings
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for saving
        data = {
            'embeddings': [result.embedding for result in results],
            'chunk_ids': [result.chunk_id for result in results],
            'metadata': [result.metadata for result in results]
        }
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(results)} embeddings to {output_path}")
    
    def load_embeddings(self, input_path: str) -> List[EmbeddingResult]:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to load embeddings from
            
        Returns:
            List of EmbeddingResult objects
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        results = []
        for embedding, chunk_id, metadata in zip(data['embeddings'], 
                                                data['chunk_ids'], 
                                                data['metadata']):
            result = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata=metadata
            )
            results.append(result)
        
        print(f"Loaded {len(results)} embeddings from {input_path}")
        return results


class EmbeddingRetriever:
    """Retriever for finding similar embeddings."""
    
    def __init__(self, embeddings: List[EmbeddingResult]):
        """
        Initialize the retriever.
        
        Args:
            embeddings: List of EmbeddingResult objects
        """
        self.embeddings = embeddings
        self.embedding_matrix = np.array([result.embedding for result in embeddings])
        self.chunk_ids = [result.chunk_id for result in embeddings]
        self.metadata = [result.metadata for result in embeddings]
    
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         top_k: int = 5,
                         threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        # Compute cosine similarities
        similarities = np.dot(self.embedding_matrix, query_embedding) / (
            np.linalg.norm(self.embedding_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((
                    self.chunk_ids[idx],
                    float(similarities[idx]),
                    self.metadata[idx]
                ))
        
        return results


def main():
    """Test the embedding pipeline."""
    # Create sample chunks
    from .text_chunker import TextChunk
    
    sample_chunks = [
        TextChunk(
            text="This is a sample text about machine learning and neural networks.",
            chunk_id="test_001_0",
            arxiv_id="test_001",
            section_title="Introduction",
            section_type="introduction",
            chunk_index=0,
            total_chunks=1,
            token_count=15,
            metadata={}
        ),
        TextChunk(
            text="Another sample text discussing transformers and attention mechanisms.",
            chunk_id="test_001_1",
            arxiv_id="test_001",
            section_title="Method",
            section_type="method",
            chunk_index=1,
            total_chunks=1,
            token_count=12,
            metadata={}
        )
    ]
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    
    # Process chunks
    results = pipeline.process_chunks(sample_chunks)
    
    print(f"Generated {len(results)} embeddings")
    for result in results:
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Embedding shape: {result.embedding.shape}")
        print(f"Metadata: {result.metadata}")
        print()
    
    # Test retrieval
    retriever = EmbeddingRetriever(results)
    
    # Create a query
    query_text = "machine learning neural networks"
    query_embedding = pipeline.generate_embedding(query_text)
    
    # Search for similar chunks
    similar_chunks = retriever.similarity_search(query_embedding, top_k=2)
    
    print("Similar chunks:")
    for chunk_id, similarity, metadata in similar_chunks:
        print(f"Chunk ID: {chunk_id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Metadata: {metadata}")
        print()


if __name__ == "__main__":
    main()
