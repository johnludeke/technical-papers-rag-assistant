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
    
    def _resolve_inputs(self, content: str, base_dir: str, processed_files: set = None) -> str:
        """
        Recursively resolve \input{} and \include{} commands in LaTeX content.

        Args:
            content: LaTeX content
            base_dir: Directory containing the .tex files
            processed_files: Set of already processed files to avoid loops

        Returns:
            Content with all inputs resolved
        """
        if processed_files is None:
            processed_files = set()

        # Find all \input{} and \include{} commands
        input_pattern = r'\\(?:input|include)\{([^}]+)\}'
        matches = list(re.finditer(input_pattern, content))

        # Process matches in reverse order to maintain correct positions
        for match in reversed(matches):
            filename = match.group(1)

            # Add .tex extension if not present
            if not filename.endswith('.tex'):
                filename += '.tex'

            # Check if already processed to avoid infinite loops
            if filename in processed_files:
                continue

            # Try to find and read the file
            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        included_content = f.read()

                    # Mark as processed
                    processed_files.add(filename)

                    # Recursively resolve inputs in the included file
                    included_content = self._resolve_inputs(included_content, base_dir, processed_files)

                    # Replace the \input command with the actual content
                    content = content[:match.start()] + '\n' + included_content + '\n' + content[match.end():]
                except Exception as e:
                    print(f"Warning: Could not read included file {filename}: {e}")

        return content

    def extract_tar_gz(self, tar_path: str) -> str:
        """Extract LaTeX source from tar.gz file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(temp_dir)

                # Find the main .tex file
                tex_files = []
                main_candidates = []

                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.tex'):
                            tex_path = os.path.join(root, file)
                            file_size = os.path.getsize(tex_path)
                            tex_files.append((tex_path, file_size))

                            # Prefer common main file names
                            if file.lower() in ['main.tex', 'ms.tex', 'paper.tex']:
                                main_candidates.append((tex_path, file_size))

                if not tex_files:
                    raise ValueError("No .tex files found in archive")

                # Use main.tex/ms.tex if found, otherwise use largest file
                if main_candidates:
                    main_tex = max(main_candidates, key=lambda x: x[1])[0]
                else:
                    main_tex = max(tex_files, key=lambda x: x[1])[0]

                base_dir = os.path.dirname(main_tex)

                with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Resolve all \input{} and \include{} commands
                content = self._resolve_inputs(content, base_dir)

                return content
    
    def clean_latex(self, latex_content: str) -> str:
        """Clean LaTeX content by removing commands and formatting."""
        content = latex_content

        # Remove LaTeX comments (lines starting with %)
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)

        # Remove LaTeX commands
        for pattern, replacement in self.latex_commands.items():
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        # Remove remaining LaTeX figure/table references
        content = re.sub(r'\[scale=[^\]]*\]\{[^}]*\}', '', content)  # [scale=0.6]{Figures/...}
        content = re.sub(r'\\includegraphics[^\n]*', '', content)
        content = re.sub(r'\\caption\{[^}]*\}', '', content)

        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs
        content = re.sub(r'\n[ \t]+', '\n', content)  # Leading whitespace
        content = re.sub(r'^\s*\n', '', content, flags=re.MULTILINE)  # Empty lines

        return content.strip()
    
    def extract_sections(self, cleaned_text: str) -> List[DocumentSection]:
        """Extract document sections from cleaned text, including subsections."""
        sections = []

        # Split by section markers (## for sections, ### for subsections, #### for subsubsections)
        # Use a pattern that captures both the header level and title
        section_pattern = r'\n(#{2,4})\s+([^\n]+)\n'
        parts = re.split(section_pattern, cleaned_text)

        # parts[0] is content before first section (usually preamble)
        # Then alternates: level, title, content, level, title, content, ...
        i = 1
        while i < len(parts) - 2:
            level_markers = parts[i]  # "##" or "###" or "####"
            title = parts[i + 1].strip()
            content = parts[i + 2].strip() if i + 2 < len(parts) else ""

            # Determine section type
            section_type = self._classify_section(title)

            if content:  # Only add non-empty sections
                sections.append(DocumentSection(
                    title=title,
                    content=content,
                    section_type=section_type
                ))

            i += 3

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
