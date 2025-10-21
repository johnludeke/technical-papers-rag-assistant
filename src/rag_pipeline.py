"""
Main RAG pipeline that integrates all components.
"""
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .arxiv_client import ArxivClient, PaperInfo
from .latex_parser import LatexParser, ProcessedDocument
from .text_chunker import TextChunker, TextChunk
from .embedding_pipeline import EmbeddingPipeline, EmbeddingResult
from .vector_store import FAISSVectorStore, VectorStoreResult


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    query: str
    retrieved_chunks: List[VectorStoreResult]
    context: str
    sources: List[Dict]


class RAGPipeline:
    """Main RAG pipeline for document retrieval and question answering."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 vector_store_dir: str = "data/vector_stores",
                 model_name: str = "google/gemma-2-2b-it"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory for storing downloaded papers
            vector_store_dir: Directory for vector stores
            model_name: Name of the embedding model
        """
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        
        # Initialize components
        self.arxiv_client = ArxivClient(os.path.join(data_dir, "papers"))
        self.latex_parser = LatexParser()
        self.text_chunker = TextChunker()
        self.embedding_pipeline = EmbeddingPipeline(model_name=model_name)
        self.vector_store = None
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)
    
    def download_papers(self, 
                       query: str = "transformer attention",
                       max_papers: int = 15) -> List[PaperInfo]:
        """
        Download papers from arXiv.
        
        Args:
            query: Search query for papers
            max_papers: Maximum number of papers to download
            
        Returns:
            List of downloaded paper information
        """
        print(f"Searching for papers with query: '{query}'")
        papers = self.arxiv_client.search_ml_papers(query=query, max_results=max_papers)
        
        print(f"Found {len(papers)} papers, downloading...")
        downloaded_papers = []
        
        for paper in tqdm(papers, desc="Downloading papers"):
            try:
                files = self.arxiv_client.download_paper(paper)
                if 'latex' in files:  # Only process papers with LaTeX source
                    downloaded_papers.append(paper)
                    print(f"✓ Downloaded: {paper.title[:50]}...")
                else:
                    print(f"✗ No LaTeX source: {paper.title[:50]}...")
            except Exception as e:
                print(f"✗ Failed to download {paper.title[:50]}: {e}")
        
        print(f"Successfully downloaded {len(downloaded_papers)} papers with LaTeX sources")
        return downloaded_papers
    
    def process_papers(self, papers: List[PaperInfo]) -> List[ProcessedDocument]:
        """
        Process downloaded papers.
        
        Args:
            papers: List of downloaded paper information
            
        Returns:
            List of processed documents
        """
        print(f"Processing {len(papers)} papers...")
        processed_docs = []
        
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                # Get paths
                paper_dir = os.path.join(self.data_dir, "papers", paper.arxiv_id)
                tar_path = os.path.join(paper_dir, f"{paper.arxiv_id}.tar.gz")
                metadata_path = os.path.join(paper_dir, "metadata.json")
                
                if os.path.exists(tar_path):
                    # Process the document
                    doc = self.latex_parser.process_document(tar_path, metadata_path)
                    processed_docs.append(doc)
                    print(f"✓ Processed: {doc.title[:50]}...")
                else:
                    print(f"✗ LaTeX source not found for {paper.arxiv_id}")
                    
            except Exception as e:
                print(f"✗ Failed to process {paper.arxiv_id}: {e}")
        
        print(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def create_chunks(self, processed_docs: List[ProcessedDocument]) -> List[TextChunk]:
        """
        Create text chunks from processed documents.
        
        Args:
            processed_docs: List of processed documents
            
        Returns:
            List of text chunks
        """
        print(f"Creating chunks from {len(processed_docs)} documents...")
        all_chunks = []
        
        for doc in tqdm(processed_docs, desc="Chunking documents"):
            try:
                chunks = self.text_chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                print(f"✓ Created {len(chunks)} chunks from {doc.title[:50]}...")
            except Exception as e:
                print(f"✗ Failed to chunk {doc.arxiv_id}: {e}")
        
        print(f"Created {len(all_chunks)} total chunks")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[EmbeddingResult]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding results
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embedding_results = self.embedding_pipeline.process_chunks(chunks)
        print(f"Generated {len(embedding_results)} embeddings")
        return embedding_results
    
    def build_vector_store(self, 
                          embedding_results: List[EmbeddingResult],
                          chunks: List[TextChunk],
                          store_name: str = "ml_papers") -> FAISSVectorStore:
        """
        Build the vector store from embeddings.
        
        Args:
            embedding_results: List of embedding results
            chunks: List of text chunks
            store_name: Name of the vector store
            
        Returns:
            Built FAISS vector store
        """
        print(f"Building vector store '{store_name}'...")
        
        # Create vector store
        self.vector_store = FAISSVectorStore(dimension=embedding_results[0].embedding.shape[0])
        
        # Prepare data
        embeddings = [result.embedding for result in embedding_results]
        chunk_ids = [result.chunk_id for result in embedding_results]
        texts = [chunk.text for chunk in chunks]
        metadata = [result.metadata for result in embedding_results]
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, chunk_ids, texts, metadata)
        
        # Save vector store
        store_dir = os.path.join(self.vector_store_dir, store_name)
        self.vector_store.save(store_dir)
        
        print(f"Vector store built and saved with {self.vector_store.get_stats()['total_chunks']} chunks")
        return self.vector_store
    
    def load_vector_store(self, store_name: str = "ml_papers") -> FAISSVectorStore:
        """
        Load an existing vector store.
        
        Args:
            store_name: Name of the vector store to load
            
        Returns:
            Loaded FAISS vector store
        """
        store_dir = os.path.join(self.vector_store_dir, store_name)
        self.vector_store = FAISSVectorStore()
        self.vector_store.load(store_dir)
        print(f"Loaded vector store with {self.vector_store.get_stats()['total_chunks']} chunks")
        return self.vector_store
    
    def query(self, 
              query: str,
              top_k: int = 5,
              score_threshold: float = 0.0) -> RAGResult:
        """
        Query the RAG system.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity threshold
            
        Returns:
            RAG result with retrieved chunks and context
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_pipeline.generate_embedding(query)
        
        # Search vector store
        retrieved_chunks = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            score_threshold=score_threshold
        )
        
        # Build context
        context_parts = []
        sources = []
        
        for chunk in retrieved_chunks:
            context_parts.append(f"[{chunk.chunk_id}] {chunk.text}")
            sources.append({
                'chunk_id': chunk.chunk_id,
                'score': chunk.score,
                'metadata': chunk.metadata
            })
        
        context = "\n\n".join(context_parts)
        
        return RAGResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            context=context,
            sources=sources
        )
    
    def build_complete_pipeline(self, 
                               query: str = "transformer attention",
                               max_papers: int = 15,
                               store_name: str = "ml_papers") -> FAISSVectorStore:
        """
        Build the complete RAG pipeline from scratch.
        
        Args:
            query: Search query for papers
            max_papers: Maximum number of papers to download
            store_name: Name of the vector store
            
        Returns:
            Built FAISS vector store
        """
        print("Building complete RAG pipeline...")
        
        # Step 1: Download papers
        papers = self.download_papers(query=query, max_papers=max_papers)
        
        # Step 2: Process papers
        processed_docs = self.process_papers(papers)
        
        # Step 3: Create chunks
        chunks = self.create_chunks(processed_docs)
        
        # Step 4: Generate embeddings
        embedding_results = self.generate_embeddings(chunks)
        
        # Step 5: Build vector store
        vector_store = self.build_vector_store(embedding_results, chunks, store_name)
        
        print("Complete RAG pipeline built successfully!")
        return vector_store


def main():
    """Test the complete RAG pipeline."""
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Build pipeline (this will take a while)
    print("Building RAG pipeline...")
    vector_store = pipeline.build_complete_pipeline(
        query="transformer attention",
        max_papers=5,  # Small number for testing
        store_name="test_ml_papers"
    )
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How do transformers work?",
        "What are the advantages of self-attention?",
        "How is positional encoding used?",
        "What is the multi-head attention?"
    ]
    
    print("\n" + "="*60)
    print("TESTING RAG PIPELINE")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = pipeline.query(query, top_k=3)
        
        print(f"Retrieved {len(result.retrieved_chunks)} chunks:")
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            print(f"\n{i}. Score: {chunk.score:.4f}")
            print(f"   Chunk ID: {chunk.chunk_id}")
            print(f"   Section: {chunk.metadata.get('section_title', 'Unknown')}")
            print(f"   Text: {chunk.text[:200]}...")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
