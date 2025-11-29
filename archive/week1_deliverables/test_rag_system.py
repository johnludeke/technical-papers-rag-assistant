"""
Test script for the RAG system.
"""
import os
import sys
from src.rag_pipeline import RAGPipeline


def test_small_pipeline():
    """Test with a small number of papers."""
    print("🚀 Testing RAG System with Small Dataset")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Build pipeline with just 3 papers for testing
    print("Building pipeline with 3 papers...")
    vector_store = pipeline.build_complete_pipeline(
        query="transformer attention",
        max_papers=3,
        store_name="test_small"
    )
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How do transformers work?",
        "What is self-attention?",
        "How is positional encoding implemented?",
        "What are the benefits of multi-head attention?"
    ]
    
    print("\n" + "=" * 60)
    print("🔍 TESTING RETRIEVAL")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        result = pipeline.query(query, top_k=2)
        
        print(f"Retrieved {len(result.retrieved_chunks)} chunks:")
        for j, chunk in enumerate(result.retrieved_chunks, 1):
            print(f"\n   {j}. Score: {chunk.score:.4f}")
            print(f"      Paper: {chunk.metadata.get('arxiv_id', 'Unknown')}")
            print(f"      Section: {chunk.metadata.get('section_title', 'Unknown')}")
            print(f"      Text: {chunk.text[:150]}...")
        
        print("\n" + "-" * 40)
    
    print("\n✅ Test completed successfully!")
    return vector_store


def test_needle_in_haystack():
    """Test needle-in-a-haystack retrieval."""
    print("\n🎯 Testing Needle-in-a-Haystack Retrieval")
    print("=" * 50)
    
    pipeline = RAGPipeline()
    
    # Load existing vector store
    try:
        vector_store = pipeline.load_vector_store("test_small")
    except:
        print("No existing vector store found. Building new one...")
        vector_store = pipeline.build_complete_pipeline(
            query="transformer attention",
            max_papers=3,
            store_name="test_small"
        )
    
    # Test specific details that should be in the papers
    needle_queries = [
        "What is the dimension of the key vectors?",
        "How many attention heads are used?",
        "What is the dropout rate?",
        "What activation function is used?",
        "What is the learning rate?",
        "How many layers are in the model?",
        "What is the batch size?",
        "What optimizer is used?"
    ]
    
    print("\nSearching for specific details...")
    for query in needle_queries:
        print(f"\nQuery: {query}")
        result = pipeline.query(query, top_k=1, score_threshold=0.1)
        
        if result.retrieved_chunks:
            chunk = result.retrieved_chunks[0]
            print(f"✓ Found (Score: {chunk.score:.4f}): {chunk.text[:100]}...")
        else:
            print("✗ No relevant chunks found")
    
    print("\n✅ Needle-in-a-haystack test completed!")


def main():
    """Main test function."""
    print("🧪 RAG System Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Small pipeline
        test_small_pipeline()
        
        # Test 2: Needle in haystack
        test_needle_in_haystack()
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Try increasing the number of papers (max_papers=15)")
        print("2. Test with different queries")
        print("3. Experiment with different embedding models")
        print("4. Build the Week 2 RAG pipeline with LLM integration")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
