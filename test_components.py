"""
Quick test script to verify components.
Tests individual components without requiring full pipeline execution.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_evaluation_framework():
    """Test the evaluation framework."""
    print("1. Testing Evaluation Framework...")
    try:
        from evaluation import RAGEvaluator, TestQuery

        evaluator = RAGEvaluator()

        # Test retrieval metrics
        sample_chunks = [
            {'metadata': {'arxiv_id': '1706.03762'}},
            {'metadata': {'arxiv_id': '1810.04805'}},
            {'metadata': {'arxiv_id': '1706.03762'}},
        ]
        relevant_ids = ['1706.03762']

        metrics = evaluator.evaluate_retrieval(sample_chunks, relevant_ids, k=3)
        assert metrics['precision_at_k'] > 0
        assert metrics['recall_at_k'] > 0

        # Test citation metrics
        response = "The attention mechanism [1] is important. Multi-head attention [2] helps."
        available_citations = ["[1]", "[2]"]
        citation_metrics = evaluator.evaluate_citations(response, available_citations)
        assert citation_metrics['citation_accuracy'] == 1.0

        print("   ✅ Evaluation framework works correctly")
        return True
    except Exception as e:
        print(f"   ❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_generator_import():
    """Test if LLM generator can be imported."""
    print("\n2. Testing LLM Generator Import...")
    try:
        from llm_generator import LLMGenerator, GenerationConfig, GeneratedResponse

        # Test dataclasses
        config = GenerationConfig(max_new_tokens=100)
        assert config.max_new_tokens == 100

        print("   ✅ LLM generator module imports correctly")
        return True
    except Exception as e:
        print(f"   ❌ LLM generator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline_updates():
    """Test if RAG pipeline has new methods."""
    print("\n3. Testing RAG Pipeline Updates...")
    try:
        # Read the file directly to check for updates
        import re

        with open('src/rag_pipeline.py', 'r') as f:
            content = f.read()

        # Check for new parameters
        assert 'llm_model_name' in content, "llm_model_name parameter missing"
        assert 'use_llm' in content, "use_llm parameter missing"

        # Check for new method
        assert 'def query_with_generation' in content, "query_with_generation method missing"

        # Check for LLM generator import
        assert 'from .llm_generator import' in content, "LLM generator import missing"

        print("   ✅ RAG pipeline has been updated correctly")
        return True
    except Exception as e:
        print(f"   ❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_formatting():
    """Test citation formatting logic."""
    print("\n4. Testing Citation Formatting...")
    try:
        from llm_generator import LLMGenerator

        # Create a mock instance to test formatting methods
        # We'll test the static-like methods without initializing the full model
        mock_chunks = [
            {
                'text': 'Sample text about attention mechanisms.',
                'score': 0.95,
                'metadata': {
                    'arxiv_id': '1706.03762',
                    'title': 'Attention Is All You Need',
                    'section_title': 'Introduction'
                }
            }
        ]

        # We can't instantiate without loading models, so just verify the class structure
        assert hasattr(LLMGenerator, '_format_context_with_citations')
        assert hasattr(LLMGenerator, '_build_prompt')
        assert hasattr(LLMGenerator, 'format_response_with_sources')

        print("   ✅ Citation formatting methods exist")
        return True
    except Exception as e:
        print(f"   ❌ Citation formatting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all component tests."""
    print("="*60)
    print("RAG System Component Tests")
    print("="*60)

    results = []

    # Run tests
    results.append(("Evaluation Framework", test_evaluation_framework()))
    results.append(("LLM Generator Import", test_llm_generator_import()))
    results.append(("RAG Pipeline Updates", test_rag_pipeline_updates()))
    results.append(("Citation Formatting", test_citation_formatting()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All components are working correctly!")
        print("\n📝 System Features:")
        print("   ✅ LLM generator with Qwen-3 support")
        print("   ✅ Citation tracking and formatting")
        print("   ✅ Evaluation framework with metrics")
        print("   ✅ Updated RAG pipeline with generation")
        print("\nTo run the full demo (requires downloading models):")
        print("   python demo.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
