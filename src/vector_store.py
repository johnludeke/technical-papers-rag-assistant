"""
FAISS vector store for storing and retrieving embeddings.
"""
import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json
from tqdm import tqdm


@dataclass
class VectorStoreResult:
    """Result from vector store search."""
    chunk_id: str
    score: float
    metadata: Dict
    text: str


class FAISSVectorStore:
    """FAISS-based vector store for embeddings."""
    
    def __init__(self, 
                 dimension: int = 384,  # Default for all-MiniLM-L6-v2
                 index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata_store = {}
        self.text_store = {}
        self.chunk_ids = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index."""
        if self.index_type == "flat":
            # Flat index for exact search
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # IVF index for approximate search
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            # HNSW index for approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_embeddings(self, 
                      embeddings: List[np.ndarray],
                      chunk_ids: List[str],
                      texts: List[str],
                      metadata: List[Dict]):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            chunk_ids: List of chunk IDs
            texts: List of text content
            metadata: List of metadata dictionaries
        """
        if len(embeddings) != len(chunk_ids) != len(texts) != len(metadata):
            raise ValueError("All input lists must have the same length")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = []
        for embedding in embeddings:
            normalized = embedding / np.linalg.norm(embedding)
            normalized_embeddings.append(normalized)
        
        embeddings_array = np.array(normalized_embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata and text
        for chunk_id, text, meta in zip(chunk_ids, texts, metadata):
            self.metadata_store[chunk_id] = meta
            self.text_store[chunk_id] = text
            self.chunk_ids.append(chunk_id)
        
        print(f"Added {len(embeddings)} embeddings to vector store")
    
    def search(self, 
               query_embedding: np.ndarray,
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[VectorStoreResult]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity threshold
            
        Returns:
            List of VectorStoreResult objects
        """
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            if score >= score_threshold:
                chunk_id = self.chunk_ids[idx]
                result = VectorStoreResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    metadata=self.metadata_store[chunk_id],
                    text=self.text_store[chunk_id]
                )
                results.append(result)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[VectorStoreResult]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
            
        Returns:
            VectorStoreResult if found, None otherwise
        """
        if chunk_id not in self.metadata_store:
            return None
        
        return VectorStoreResult(
            chunk_id=chunk_id,
            score=1.0,  # Perfect match
            metadata=self.metadata_store[chunk_id],
            text=self.text_store[chunk_id]
        )
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.chunk_ids),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def save(self, output_dir: str):
        """
        Save the vector store to disk.
        
        Args:
            output_dir: Directory to save the vector store
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata and text stores
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        
        text_path = os.path.join(output_dir, "texts.pkl")
        with open(text_path, 'wb') as f:
            pickle.dump(self.text_store, f)
        
        # Save chunk IDs
        chunk_ids_path = os.path.join(output_dir, "chunk_ids.pkl")
        with open(chunk_ids_path, 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_chunks': len(self.chunk_ids)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Vector store saved to {output_dir}")
    
    def load(self, input_dir: str):
        """
        Load the vector store from disk.
        
        Args:
            input_dir: Directory to load the vector store from
        """
        # Load FAISS index
        index_path = os.path.join(input_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        
        # Load metadata and text stores
        metadata_path = os.path.join(input_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        text_path = os.path.join(input_dir, "texts.pkl")
        with open(text_path, 'rb') as f:
            self.text_store = pickle.load(f)
        
        chunk_ids_path = os.path.join(input_dir, "chunk_ids.pkl")
        with open(chunk_ids_path, 'rb') as f:
            self.chunk_ids = pickle.load(f)
        
        # Load configuration
        config_path = os.path.join(input_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.dimension = config['dimension']
        self.index_type = config['index_type']
        
        print(f"Vector store loaded from {input_dir}")
        print(f"Loaded {len(self.chunk_ids)} chunks")


class VectorStoreManager:
    """Manager for multiple vector stores."""
    
    def __init__(self, base_dir: str = "data/vector_stores"):
        """
        Initialize the vector store manager.
        
        Args:
            base_dir: Base directory for vector stores
        """
        self.base_dir = base_dir
        self.stores = {}
        os.makedirs(base_dir, exist_ok=True)
    
    def create_store(self, 
                    store_name: str,
                    dimension: int = 384,
                    index_type: str = "flat") -> FAISSVectorStore:
        """
        Create a new vector store.
        
        Args:
            store_name: Name of the vector store
            dimension: Dimension of embeddings
            index_type: Type of FAISS index
            
        Returns:
            Created FAISSVectorStore
        """
        store = FAISSVectorStore(dimension=dimension, index_type=index_type)
        self.stores[store_name] = store
        return store
    
    def load_store(self, store_name: str) -> FAISSVectorStore:
        """
        Load an existing vector store.
        
        Args:
            store_name: Name of the vector store to load
            
        Returns:
            Loaded FAISSVectorStore
        """
        store_dir = os.path.join(self.base_dir, store_name)
        if not os.path.exists(store_dir):
            raise FileNotFoundError(f"Vector store {store_name} not found")
        
        store = FAISSVectorStore()
        store.load(store_dir)
        self.stores[store_name] = store
        return store
    
    def save_store(self, store_name: str):
        """
        Save a vector store.
        
        Args:
            store_name: Name of the vector store to save
        """
        if store_name not in self.stores:
            raise ValueError(f"Vector store {store_name} not found")
        
        store_dir = os.path.join(self.base_dir, store_name)
        self.stores[store_name].save(store_dir)
    
    def list_stores(self) -> List[str]:
        """List available vector stores."""
        return os.listdir(self.base_dir)


def main():
    """Test the vector store."""
    # Create sample embeddings
    dimension = 384
    num_embeddings = 100
    
    embeddings = [np.random.rand(dimension) for _ in range(num_embeddings)]
    chunk_ids = [f"chunk_{i:03d}" for i in range(num_embeddings)]
    texts = [f"This is sample text for chunk {i}" for i in range(num_embeddings)]
    metadata = [{"chunk_index": i, "section": "test"} for i in range(num_embeddings)]
    
    # Create vector store
    store = FAISSVectorStore(dimension=dimension)
    store.add_embeddings(embeddings, chunk_ids, texts, metadata)
    
    print(f"Created vector store with {store.get_stats()['total_chunks']} chunks")
    
    # Test search
    query_embedding = np.random.rand(dimension)
    results = store.search(query_embedding, top_k=5)
    
    print(f"\nSearch results:")
    for result in results:
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.text}")
        print()
    
    # Test save/load
    store.save("test_vector_store")
    
    # Load new store
    new_store = FAISSVectorStore()
    new_store.load("test_vector_store")
    
    print(f"Loaded vector store with {new_store.get_stats()['total_chunks']} chunks")
    
    # Clean up
    import shutil
    shutil.rmtree("test_vector_store")


if __name__ == "__main__":
    main()
