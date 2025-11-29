"""
Simple test script for the RAG system components.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from arxiv_client import ArxivClient
        print("✅ arxiv_client imported successfully")
    except Exception as e:
        print(f"❌ arxiv_client failed: {e}")
    
    try:
        from latex_parser import LatexParser
        print("✅ latex_parser imported successfully")
    except Exception as e:
        print(f"❌ latex_parser failed: {e}")
    
    try:
        from text_chunker import TextChunker
        print("✅ text_chunker imported successfully")
    except Exception as e:
        print(f"❌ text_chunker failed: {e}")
    
    try:
        from embedding_pipeline import EmbeddingPipeline
        print("✅ embedding_pipeline imported successfully")
    except Exception as e:
        print(f"❌ embedding_pipeline failed: {e}")
    
    try:
        from vector_store import FAISSVectorStore
        print("✅ vector_store imported successfully")
    except Exception as e:
        print(f"❌ vector_store failed: {e}")

def test_arxiv_client():
    """Test the arXiv client."""
    print("\n🔍 Testing arXiv client...")
    
    try:
        from arxiv_client import ArxivClient
        
        client = ArxivClient()
        papers = client.search_ml_papers(query="transformer", max_results=3)
        
        print(f"✅ Found {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"   {i}. {paper.title[:50]}...")
        
        return papers
    except Exception as e:
        print(f"❌ arXiv client test failed: {e}")
        return []

def test_latex_parser():
    """Test the LaTeX parser."""
    print("\n📄 Testing LaTeX parser...")
    
    try:
        from latex_parser import LatexParser
        
        parser = LatexParser()
        
        # Test with sample LaTeX
        sample_latex = """
        \\documentclass{article}
        \\title{Test Paper}
        \\begin{document}
        \\section{Introduction}
        This is a test introduction.
        \\section{Method}
        This is the method section.
        \\end{document}
        """
        
        cleaned = parser.clean_latex(sample_latex)
        sections = parser.extract_sections(cleaned)
        
        print(f"✅ LaTeX parser working, found {len(sections)} sections")
        for section in sections:
            print(f"   - {section.title}: {section.section_type}")
        
        return True
    except Exception as e:
        print(f"❌ LaTeX parser test failed: {e}")
        return False

def test_text_chunker():
    """Test the text chunker."""
    print("\n✂️ Testing text chunker...")
    
    try:
        from text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=50, overlap_size=10)
        
        test_text = "This is a test sentence. " * 10  # Create longer text
        
        chunks = chunker.chunk_text(
            test_text,
            arxiv_id="test_001",
            section_title="Test Section",
            section_type="test"
        )
        
        print(f"✅ Text chunker working, created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"   {i}. {chunk.chunk_id} ({chunk.token_count} tokens)")
        
        return True
    except Exception as e:
        print(f"❌ Text chunker test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG System Component Tests")
    print("=" * 50)
    
    # Test imports
    test_imports()
    
    # Test components
    test_arxiv_client()
    test_latex_parser()
    test_text_chunker()
    
    print("\n🎉 Component tests completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run full pipeline test")
    print("3. Download and process actual papers")

if __name__ == "__main__":
    main()
