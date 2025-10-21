"""
LaTeX text extraction and cleaning pipeline.
"""
import os
import re
import tarfile
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class DocumentSection:
    """Represents a section of a document."""
    title: str
    content: str
    section_type: str  # 'abstract', 'introduction', 'method', 'results', 'conclusion', etc.
    page_number: Optional[int] = None
    line_number: Optional[int] = None


@dataclass
class ProcessedDocument:
    """Represents a processed LaTeX document."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[DocumentSection]
    metadata: Dict
    raw_text: str


class LatexParser:
    """Parser for extracting and cleaning text from LaTeX sources."""
    
    def __init__(self):
        # Common LaTeX commands to remove or replace
        self.latex_commands = {
            r'\\cite\{[^}]*\}': '',  # Citations
            r'\\ref\{[^}]*\}': '',  # References
            r'\\label\{[^}]*\}': '',  # Labels
            r'\\footnote\{[^}]*\}': '',  # Footnotes
            r'\\url\{[^}]*\}': '',  # URLs
            r'\\href\{[^}]*\}\{[^}]*\}': '',  # Hyperlinks
            r'\\textbf\{([^}]*)\}': r'\1',  # Bold text
            r'\\textit\{([^}]*)\}': r'\1',  # Italic text
            r'\\emph\{([^}]*)\}': r'\1',  # Emphasis
            r'\\textsc\{([^}]*)\}': r'\1',  # Small caps
            r'\\texttt\{([^}]*)\}': r'\1',  # Typewriter
            r'\\text\{([^}]*)\}': r'\1',  # Text mode
            r'\\math[A-Za-z]*\{[^}]*\}': '',  # Math commands
            r'\$[^$]*\$': '',  # Inline math
            r'\$\$[^$]*\$\$': '',  # Display math
            r'\\begin\{equation\}[^\\]*\\end\{equation\}': '',  # Equations
            r'\\begin\{align\}[^\\]*\\end\{align\}': '',  # Align environments
            r'\\begin\{figure\}[^\\]*\\end\{figure\}': '',  # Figures
            r'\\begin\{table\}[^\\]*\\end\{table\}': '',  # Tables
            r'\\begin\{algorithm\}[^\\]*\\end\{algorithm\}': '',  # Algorithms
            r'\\begin\{itemize\}[^\\]*\\end\{itemize\}': '',  # Itemize
            r'\\begin\{enumerate\}[^\\]*\\end\{enumerate\}': '',  # Enumerate
            r'\\item\s*': '• ',  # List items
            r'\\section\{([^}]*)\}': r'\n\n## \1\n\n',  # Sections
            r'\\subsection\{([^}]*)\}': r'\n\n### \1\n\n',  # Subsections
            r'\\subsubsection\{([^}]*)\}': r'\n\n#### \1\n\n',  # Subsubsections
            r'\\paragraph\{([^}]*)\}': r'\n\n**\1**\n\n',  # Paragraphs
            r'\\newline': '\n',  # New lines
            r'\\par': '\n\n',  # Paragraph breaks
            r'\\clearpage': '\n\n',  # Page breaks
            r'\\newpage': '\n\n',  # Page breaks
            r'\\linebreak': '\n',  # Line breaks
            r'\\pagebreak': '\n\n',  # Page breaks
            r'\\\\': '\n',  # Line breaks
            r'\\[A-Za-z]+\{[^}]*\}': '',  # Generic LaTeX commands
            r'\\[A-Za-z]+': '',  # Generic LaTeX commands without braces
        }
        
        # Section patterns for identifying document structure
        self.section_patterns = {
            'abstract': [r'\\begin\{abstract\}', r'\\abstract'],
            'introduction': [r'\\section\{.*[Ii]ntroduction.*\}', r'\\section\{.*[Ii]ntro.*\}'],
            'method': [r'\\section\{.*[Mm]ethod.*\}', r'\\section\{.*[Aa]pproach.*\}', r'\\section\{.*[Dd]esign.*\}'],
            'results': [r'\\section\{.*[Rr]esult.*\}', r'\\section\{.*[Ee]xperiment.*\}'],
            'conclusion': [r'\\section\{.*[Cc]onclusion.*\}', r'\\section\{.*[Dd]iscussion.*\}'],
            'related_work': [r'\\section\{.*[Rr]elated.*\}', r'\\section\{.*[Ll]iterature.*\}'],
            'background': [r'\\section\{.*[Bb]ackground.*\}', r'\\section\{.*[Pp]reliminaries.*\}']
        }
    
    def extract_tar_gz(self, tar_path: str) -> str:
        """Extract LaTeX source from tar.gz file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
                
                # Find the main .tex file (usually the largest or most complex)
                tex_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.tex'):
                            tex_path = os.path.join(root, file)
                            tex_files.append((tex_path, os.path.getsize(tex_path)))
                
                if not tex_files:
                    raise ValueError("No .tex files found in archive")
                
                # Get the largest .tex file (likely the main document)
                main_tex = max(tex_files, key=lambda x: x[1])[0]
                
                with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                return content
    
    def clean_latex(self, latex_content: str) -> str:
        """Clean LaTeX content by removing commands and formatting."""
        content = latex_content
        
        # Remove LaTeX commands
        for pattern, replacement in self.latex_commands.items():
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs
        content = re.sub(r'\n[ \t]+', '\n', content)  # Leading whitespace
        
        return content.strip()
    
    def extract_sections(self, cleaned_text: str) -> List[DocumentSection]:
        """Extract document sections from cleaned text."""
        sections = []
        
        # Split by section markers
        section_parts = re.split(r'\n##\s+([^\n]+)\n', cleaned_text)
        
        for i in range(1, len(section_parts), 2):
            if i + 1 < len(section_parts):
                title = section_parts[i].strip()
                content = section_parts[i + 1].strip()
                
                # Determine section type
                section_type = self._classify_section(title)
                
                if content:  # Only add non-empty sections
                    sections.append(DocumentSection(
                        title=title,
                        content=content,
                        section_type=section_type
                    ))
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify a section based on its title."""
        title_lower = title.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return section_type
        
        return 'other'
    
    def extract_metadata(self, latex_content: str, metadata_file: str) -> Dict:
        """Extract metadata from LaTeX and JSON files."""
        metadata = {}
        
        # Try to load existing metadata
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Extract title from LaTeX if not available
        if 'title' not in metadata:
            title_match = re.search(r'\\title\{([^}]*)\}', latex_content)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
        
        # Extract authors from LaTeX if not available
        if 'authors' not in metadata:
            author_match = re.search(r'\\author\{([^}]*)\}', latex_content)
            if author_match:
                authors_text = author_match.group(1)
                # Simple author extraction (could be improved)
                authors = [author.strip() for author in re.split(r'\s+and\s+', authors_text)]
                metadata['authors'] = authors
        
        return metadata
    
    def process_document(self, tar_path: str, metadata_file: str) -> ProcessedDocument:
        """Process a complete LaTeX document."""
        # Extract LaTeX content
        latex_content = self.extract_tar_gz(tar_path)
        
        # Clean the content
        cleaned_text = self.clean_latex(latex_content)
        
        # Extract sections
        sections = self.extract_sections(cleaned_text)
        
        # Extract metadata
        metadata = self.extract_metadata(latex_content, metadata_file)
        
        # Get arxiv_id from path
        arxiv_id = os.path.basename(os.path.dirname(tar_path))
        
        return ProcessedDocument(
            arxiv_id=arxiv_id,
            title=metadata.get('title', 'Unknown Title'),
            authors=metadata.get('authors', []),
            abstract=metadata.get('abstract', ''),
            sections=sections,
            metadata=metadata,
            raw_text=cleaned_text
        )


def main():
    """Test the LaTeX parser."""
    parser = LatexParser()
    
    # Test with a sample LaTeX content
    sample_latex = """
    \\documentclass{article}
    \\title{Test Paper}
    \\author{John Doe and Jane Smith}
    \\begin{document}
    \\maketitle
    
    \\begin{abstract}
    This is a test abstract for our paper.
    \\end{abstract}
    
    \\section{Introduction}
    This is the introduction section with some text.
    
    \\section{Method}
    Here we describe our method.
    
    \\section{Results}
    Our results are presented here.
    
    \\section{Conclusion}
    We conclude with some final thoughts.
    \\end{document}
    """
    
    # Clean the content
    cleaned = parser.clean_latex(sample_latex)
    print("Cleaned content:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Extract sections
    sections = parser.extract_sections(cleaned)
    print("Extracted sections:")
    for section in sections:
        print(f"Title: {section.title}")
        print(f"Type: {section.section_type}")
        print(f"Content: {section.content[:100]}...")
        print()


if __name__ == "__main__":
    main()
