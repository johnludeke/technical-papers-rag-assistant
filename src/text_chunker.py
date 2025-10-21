"""
Text chunking strategy for processing documents into embeddable chunks.
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tiktoken


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_id: str
    arxiv_id: str
    section_title: str
    section_type: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: Dict


class TextChunker:
    """Text chunker for creating overlapping chunks of text."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap_size: int = 50,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.model_name = model_name
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # Enhanced sentence splitting that handles academic text
        sentences = []
        
        # Split on sentence boundaries
        sentence_patterns = [
            r'\.\s+[A-Z]',  # Period followed by capital letter
            r'\.\s+[0-9]',  # Period followed by number
            r'!\s+[A-Z]',   # Exclamation followed by capital letter
            r'\?\s+[A-Z]',  # Question followed by capital letter
            r'\.\s*$',      # Period at end of text
            r'!\s*$',       # Exclamation at end of text
            r'\?\s*$',      # Question at end of text
        ]
        
        # Combine patterns
        pattern = '|'.join(f'({p})' for p in sentence_patterns)
        
        # Split text
        parts = re.split(pattern, text)
        
        current_sentence = ""
        for part in parts:
            if part and not re.match(pattern, part):
                current_sentence += part
            else:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = part if part else ""
        
        # Add final sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def create_chunks_from_sentences(self, 
                                   sentences: List[str],
                                   arxiv_id: str,
                                   section_title: str,
                                   section_type: str) -> List[TextChunk]:
        """Create chunks from sentences with overlap."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, create a chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=f"{arxiv_id}_{section_type}_{chunk_index}",
                    arxiv_id=arxiv_id,
                    section_title=section_title,
                    section_type=section_type,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    token_count=current_tokens,
                    metadata={
                        'sentence_count': len(current_chunk),
                        'start_sentence': i - len(current_chunk),
                        'end_sentence': i - 1
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{arxiv_id}_{section_type}_{chunk_index}",
                arxiv_id=arxiv_id,
                section_title=section_title,
                section_type=section_type,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                token_count=current_tokens,
                metadata={
                    'sentence_count': len(current_chunk),
                    'start_sentence': len(sentences) - len(current_chunk),
                    'end_sentence': len(sentences) - 1
                }
            )
            chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap between chunks."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Add sentences from the end until we reach overlap_size tokens
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap_size:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def chunk_document(self, processed_doc) -> List[TextChunk]:
        """
        Chunk a processed document into overlapping chunks.
        
        Args:
            processed_doc: ProcessedDocument object
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        # Process each section
        for section in processed_doc.sections:
            if not section.content.strip():
                continue
            
            # Split section into sentences
            sentences = self.split_text_by_sentences(section.content)
            
            if not sentences:
                continue
            
            # Create chunks from sentences
            section_chunks = self.create_chunks_from_sentences(
                sentences,
                processed_doc.arxiv_id,
                section.title,
                section.section_type
            )
            
            all_chunks.extend(section_chunks)
        
        # Also process the abstract separately if it exists
        if processed_doc.abstract:
            abstract_sentences = self.split_text_by_sentences(processed_doc.abstract)
            if abstract_sentences:
                abstract_chunks = self.create_chunks_from_sentences(
                    abstract_sentences,
                    processed_doc.arxiv_id,
                    "Abstract",
                    "abstract"
                )
                all_chunks.extend(abstract_chunks)
        
        return all_chunks
    
    def chunk_text(self, 
                   text: str,
                   arxiv_id: str,
                   section_title: str = "Unknown",
                   section_type: str = "other") -> List[TextChunk]:
        """
        Chunk raw text into overlapping chunks.
        
        Args:
            text: Raw text to chunk
            arxiv_id: arXiv ID of the document
            section_title: Title of the section
            section_type: Type of the section
            
        Returns:
            List of TextChunk objects
        """
        sentences = self.split_text_by_sentences(text)
        return self.create_chunks_from_sentences(sentences, arxiv_id, section_title, section_type)


def main():
    """Test the text chunker."""
    chunker = TextChunker(chunk_size=100, overlap_size=20)
    
    # Test text
    test_text = """
    This is the first sentence of our test document. It contains some information about machine learning.
    This is the second sentence that continues the discussion. It talks about neural networks and their applications.
    The third sentence introduces a new concept. It discusses transformers and attention mechanisms.
    This fourth sentence provides more details. It explains how attention works in practice.
    The fifth sentence concludes this section. It summarizes the key points we've covered.
    """
    
    # Create chunks
    chunks = chunker.chunk_text(
        test_text,
        arxiv_id="test_2023.001",
        section_title="Introduction",
        section_type="introduction"
    )
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Metadata: {chunk.metadata}")


if __name__ == "__main__":
    main()
