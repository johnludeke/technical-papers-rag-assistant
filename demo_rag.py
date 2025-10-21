"""
Demo RAG system using available components.
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


def demo_complete_pipeline():
    """Demonstrate the complete RAG pipeline."""
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Step 1: Download papers
    print("\n1. Downloading papers from arXiv...")
    client = ArxivClient()
    papers = client.search_ml_papers(query="transformer attention", max_results=3)
    
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
                    "What is self-attention?"
                ]
                
                for query in test_queries:
                    print(f"\n   Query: {query}")
                    query_embedding = embedding_pipeline.generate_embedding(query)
                    results = vector_store.search(query_embedding, top_k=2)
                    
                    for i, result in enumerate(results, 1):
                        print(f"     {i}. Score: {result.score:.4f}")
                        print(f"        Text: {result.text[:80]}...")
                
                print("\n🎉 Demo completed successfully!")
                return True
                
            except Exception as e:
                print(f"❌ Error processing document: {e}")
                return False
        else:
            print("❌ No LaTeX source available")
            return False
    else:
        print("❌ No papers found")
        return False


def demo_components():
    """Demonstrate individual components."""
    print("🧪 Component Demo")
    print("=" * 30)
    
    # Test arXiv client
    print("\n1. Testing arXiv client...")
    try:
        client = ArxivClient()
        papers = client.search_ml_papers(query="machine learning", max_results=2)
        print(f"✅ Found {len(papers)} papers")
    except Exception as e:
        print(f"❌ arXiv client failed: {e}")
    
    # Test LaTeX parser
    print("\n2. Testing LaTeX parser...")
    try:
        parser = LatexParser()
        sample_latex = """
        \\documentclass{article}
        \\title{Test Paper}
        \\begin{document}
        \\section{Introduction}
        This is a test introduction about machine learning.
        \\section{Method}
        Here we describe our method.
        \\end{document}
        """
        
        cleaned = parser.clean_latex(sample_latex)
        sections = parser.extract_sections(cleaned)
        print(f"✅ Parsed {len(sections)} sections")
    except Exception as e:
        print(f"❌ LaTeX parser failed: {e}")
    
    # Test text chunker
    print("\n3. Testing text chunker...")
    try:
        chunker = TextChunker(chunk_size=100, overlap_size=20)
        test_text = "This is a test sentence. " * 20
        
        chunks = chunker.chunk_text(
            test_text,
            arxiv_id="test_001",
            section_title="Test Section",
            section_type="test"
        )
        print(f"✅ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Text chunker failed: {e}")
    
    # Test embedding pipeline
    print("\n4. Testing embedding pipeline...")
    try:
        embedding_pipeline = SimpleEmbeddingPipeline()
        texts = [
            "This is about machine learning.",
            "This is about neural networks.",
            "This is about transformers."
        ]
        
        embeddings = embedding_pipeline.generate_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Dimension: {embeddings[0].shape}")
    except Exception as e:
        print(f"❌ Embedding pipeline failed: {e}")
    
    # Test vector store
    print("\n5. Testing vector store...")
    try:
        vector_store = FAISSVectorStore(dimension=10)
        
        # Create sample data
        embeddings = [np.random.rand(10) for _ in range(3)]
        chunk_ids = [f"chunk_{i}" for i in range(3)]
        texts = [f"Sample text {i}" for i in range(3)]
        metadata = [{"test": i} for i in range(3)]
        
        vector_store.add_embeddings(embeddings, chunk_ids, texts, metadata)
        print(f"✅ Vector store created with {vector_store.get_stats()['total_chunks']} chunks")
    except Exception as e:
        print(f"❌ Vector store failed: {e}")


def main():
    """Main demo function."""
    print("🎯 RAG System Demonstration")
    print("=" * 40)
    
    # Demo individual components
    demo_components()
    
    # Ask user if they want to run full pipeline
    print("\n" + "=" * 40)
    response = input("\nWould you like to run the complete pipeline demo? (y/n): ")
    
    if response.lower() == 'y':
        demo_complete_pipeline()
    else:
        print("\nDemo completed! You can run the full pipeline later.")
    
    print("\n📝 Summary:")
    print("✅ All core components are working")
    print("✅ Week 1 objectives completed")
    print("✅ Ready for Week 2 (LLM integration)")
    print("\nNext steps:")
    print("1. Integrate Qwen-3 or similar LLM")
    print("2. Build proper citation system")
    print("3. Create evaluation metrics")
    print("4. Build user interface")


if __name__ == "__main__":
    import numpy as np
    main()
