"""
arXiv API client for downloading ML papers and LaTeX sources.
"""
import arxiv
import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class PaperInfo:
    """Information about a paper from arXiv."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    latex_url: Optional[str] = None


class ArxivClient:
    """Client for downloading papers from arXiv."""
    
    def __init__(self, download_dir: str = "data/papers"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
    def search_ml_papers(self, 
                        query: str = "machine learning",
                        max_results: int = 20,
                        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
                        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending) -> List[PaperInfo]:
        """
        Search for ML papers on arXiv.
        
        Args:
            query: Search query (default: "machine learning")
            max_results: Maximum number of papers to return
            sort_by: How to sort results
            sort_order: Sort order (descending by default)
            
        Returns:
            List of PaperInfo objects
        """
        # Create search query for ML papers
        search_query = f"cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:stat.ML AND {query}"
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        papers = []
        for result in client.results(search):
            # Extract arXiv ID from URL
            arxiv_id = result.entry_id.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '')
            
            paper_info = PaperInfo(
                arxiv_id=arxiv_id,
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                categories=result.categories,
                published=result.published.isoformat(),
                updated=result.updated.isoformat(),
                pdf_url=result.pdf_url,
                latex_url=self._get_latex_url(arxiv_id)
            )
            papers.append(paper_info)
            
        return papers
    
    def _get_latex_url(self, arxiv_id: str) -> Optional[str]:
        """Get the LaTeX source URL for a paper."""
        # arXiv LaTeX sources are typically at this URL pattern
        latex_url = f"https://arxiv.org/src/{arxiv_id}"
        return latex_url
    
    def download_paper(self, paper_info: PaperInfo) -> Dict[str, str]:
        """
        Download both PDF and LaTeX source for a paper.
        
        Args:
            paper_info: Paper information
            
        Returns:
            Dictionary with paths to downloaded files
        """
        paper_dir = os.path.join(self.download_dir, paper_info.arxiv_id)
        os.makedirs(paper_dir, exist_ok=True)
        
        downloaded_files = {}
        
        # Download PDF
        pdf_path = os.path.join(paper_dir, f"{paper_info.arxiv_id}.pdf")
        if not os.path.exists(pdf_path):
            try:
                response = requests.get(paper_info.pdf_url, stream=True)
                response.raise_for_status()
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_files['pdf'] = pdf_path
                print(f"Downloaded PDF: {paper_info.title[:50]}...")
            except Exception as e:
                print(f"Failed to download PDF for {paper_info.arxiv_id}: {e}")
        
        # Download LaTeX source
        latex_path = os.path.join(paper_dir, f"{paper_info.arxiv_id}.tar.gz")
        if not os.path.exists(latex_path) and paper_info.latex_url:
            try:
                response = requests.get(paper_info.latex_url, stream=True)
                response.raise_for_status()
                with open(latex_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_files['latex'] = latex_path
                print(f"Downloaded LaTeX: {paper_info.title[:50]}...")
            except Exception as e:
                print(f"Failed to download LaTeX for {paper_info.arxiv_id}: {e}")
        
        # Save paper metadata
        metadata_path = os.path.join(paper_dir, "metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'arxiv_id': paper_info.arxiv_id,
                'title': paper_info.title,
                'authors': paper_info.authors,
                'abstract': paper_info.abstract,
                'categories': paper_info.categories,
                'published': paper_info.published,
                'updated': paper_info.updated
            }, f, indent=2)
        
        downloaded_files['metadata'] = metadata_path
        
        # Add small delay to be respectful to arXiv servers
        time.sleep(1)
        
        return downloaded_files


def main():
    """Test the arXiv client."""
    client = ArxivClient()
    
    # Search for ML papers
    print("Searching for ML papers...")
    papers = client.search_ml_papers(query="transformer attention", max_results=5)
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Categories: {', '.join(paper.categories)}")
        print(f"   LaTeX URL: {paper.latex_url}")
        print()
    
    # Download first paper as test
    if papers:
        print(f"Downloading first paper: {papers[0].title}")
        files = client.download_paper(papers[0])
        print(f"Downloaded files: {files}")


if __name__ == "__main__":
    main()
