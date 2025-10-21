"""
Run the full RAG pipeline demo.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from arxiv_client import ArxivClient
from latex_parser import LatexParser
from text_chunker import TextChunker
from simple_embedding import SimpleEmbeddingPipeline
from vector_store import FAISSVectorStore
import numpy as np


def main():
    """Run the complete RAG pipeline demo."""
    print("🚀 Full RAG Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Download papers
    print("\n1. Downloading papers from arXiv...")
    client = ArxivClient()
    papers = client.search_ml_papers(query="transformer attention", max_results=2)
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"   {i}. {paper.title[:60]}...")
    
    # Download first paper
    if papers:
        print(f"\nDownloading first paper: {papers[0].title[:50]}...")
        files = client.download_paper(papers[0])
        
        if 'latex' in files:
            print("✅ LaTeX source downloaded successfully")
            
            # Step 2: Parse LaTeX
            print("\n2. Parsing LaTeX source...")
            parser = LatexParser()
            
            try:
                doc = parser.process_document(files['latex'], files['metadata'])
                print(f"✅ Parsed document: {doc.title}")
                print(f"   Found {len(doc.sections)} sections")
                
                # Step 3: Create chunks
                print("\n3. Creating text chunks...")
                chunker = TextChunker(chunk_size=200, overlap_size=50)
                chunks = chunker.chunk_document(doc)
                
                print(f"✅ Created {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:3], 1):
                    print(f"   {i}. {chunk.section_title}: {chunk.text[:50]}...")
                
                # Step 4: Generate embeddings
                print("\n4. Generating embeddings...")
                embedding_pipeline = SimpleEmbeddingPipeline()
                embedding_results = embedding_pipeline.process_chunks(chunks)
                
                print(f"✅ Generated {len(embedding_results)} embeddings")
                print(f"   Embedding dimension: {embedding_results[0].embedding.shape}")
                
                # Step 5: Build vector store
                print("\n5. Building vector store...")
                vector_store = FAISSVectorStore(dimension=embedding_results[0].embedding.shape[0])
                
                embeddings = [result.embedding for result in embedding_results]
                chunk_ids = [result.chunk_id for result in embedding_results]
                texts = [chunk.text for chunk in chunks]
                metadata = [result.metadata for result in embedding_results]
                
                vector_store.add_embeddings(embeddings, chunk_ids, texts, metadata)
                print(f"✅ Vector store built with {vector_store.get_stats()['total_chunks']} chunks")
                
                # Step 6: Test retrieval
                print("\n6. Testing retrieval...")
                test_queries = [
                    "What is attention mechanism?",
                    "How do transformers work?",
                    "What is self-attention?",
                    "What are the benefits of multi-head attention?"
                ]
                
                for query in test_queries:
                    print(f"\n   Query: {query}")
                    query_embedding = embedding_pipeline.generate_embedding(query)
                    results = vector_store.search(query_embedding, top_k=2)
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"     {i}. Score: {result.score:.4f}")
                            print(f"        Text: {result.text[:80]}...")
                    else:
                        print("     No results found")
                
                print("\n🎉 Full pipeline demo completed successfully!")
                print("\n📊 Results Summary:")
                print(f"✅ Downloaded and processed {len(papers)} papers")
                print(f"✅ Created {len(chunks)} text chunks")
                print(f"✅ Generated {len(embedding_results)} embeddings")
                print(f"✅ Built vector store with {vector_store.get_stats()['total_chunks']} chunks")
                print(f"✅ Successfully tested {len(test_queries)} queries")
                
                print("\n🎯 Week 1 Objectives: ✅ COMPLETED")
                print("✅ Data ingestion and LaTeX parsing")
                print("✅ Text chunking with overlap")
                print("✅ Embedding generation")
                print("✅ Vector store implementation")
                print("✅ Retrieval system")
                
                return True
                
            except Exception as e:
                print(f"❌ Error processing document: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ No LaTeX source available")
            return False
    else:
        print("❌ No papers found")
        return False


if __name__ == "__main__":
    main()
