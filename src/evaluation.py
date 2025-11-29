"""
Evaluation metrics and tests for RAG system quality.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class EvaluationMetrics:
    """Metrics for RAG system evaluation."""
    # Retrieval metrics
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float

    # Latency metrics
    avg_retrieval_time_ms: float
    avg_generation_time_ms: Optional[float]
    avg_total_time_ms: float

    # Citation metrics
    citation_accuracy: Optional[float]
    citations_per_response: Optional[float]


@dataclass
class TestQuery:
    """Test query with ground truth."""
    query: str
    relevant_doc_ids: List[str]  # List of relevant arxiv_ids
    relevant_sections: Optional[List[str]] = None  # Expected sections


class RAGEvaluator:
    """
    Evaluator for RAG system quality.
    Measures retrieval precision, recall, latency, and citation accuracy.
    """

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def evaluate_retrieval(self,
                          retrieved_chunks: List[Dict],
                          relevant_doc_ids: List[str],
                          k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            relevant_doc_ids: List of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Dictionary of retrieval metrics
        """
        # Extract retrieved doc IDs
        retrieved_doc_ids = []
        for chunk in retrieved_chunks[:k]:
            doc_id = chunk.get('metadata', {}).get('arxiv_id', None)
            if doc_id:
                retrieved_doc_ids.append(doc_id)

        # Calculate precision@k
        relevant_retrieved = len(set(retrieved_doc_ids) & set(relevant_doc_ids))
        precision = relevant_retrieved / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0

        # Calculate recall@k
        recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in relevant_doc_ids:
                mrr = 1.0 / i
                break

        # Calculate NDCG@k (Normalized Discounted Cumulative Gain)
        ndcg = self._calculate_ndcg(retrieved_doc_ids, relevant_doc_ids, k)

        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'mrr': mrr,
            'ndcg_at_k': ndcg
        }

    def _calculate_ndcg(self,
                       retrieved_doc_ids: List[str],
                       relevant_doc_ids: List[str],
                       k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Relevant document IDs
            k: Number of results to consider

        Returns:
            NDCG@k score
        """
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids[:k], 1):
            if doc_id in relevant_doc_ids:
                dcg += 1.0 / np.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant_doc_ids), k) + 1):
            idcg += 1.0 / np.log2(i + 1)

        # Normalize
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg

    def evaluate_latency(self,
                        retrieval_times: List[float],
                        generation_times: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Evaluate system latency.

        Args:
            retrieval_times: List of retrieval times in seconds
            generation_times: Optional list of generation times in seconds

        Returns:
            Dictionary of latency metrics
        """
        metrics = {
            'avg_retrieval_time_ms': np.mean(retrieval_times) * 1000,
            'p50_retrieval_time_ms': np.percentile(retrieval_times, 50) * 1000,
            'p95_retrieval_time_ms': np.percentile(retrieval_times, 95) * 1000,
        }

        if generation_times:
            metrics.update({
                'avg_generation_time_ms': np.mean(generation_times) * 1000,
                'p50_generation_time_ms': np.percentile(generation_times, 50) * 1000,
                'p95_generation_time_ms': np.percentile(generation_times, 95) * 1000,
                'avg_total_time_ms': np.mean([r + g for r, g in zip(retrieval_times, generation_times)]) * 1000,
            })

        return metrics

    def evaluate_citations(self,
                          generated_response: str,
                          available_citations: List[str]) -> Dict[str, float]:
        """
        Evaluate citation accuracy and usage.

        Args:
            generated_response: Generated response text
            available_citations: List of available citation markers (e.g., ["[1]", "[2]"])

        Returns:
            Dictionary of citation metrics
        """
        # Find all citations in response
        citation_pattern = r'\[(\d+)\]'
        found_citations = re.findall(citation_pattern, generated_response)

        # Check if citations are valid
        valid_citations = [c for c in found_citations if f"[{c}]" in available_citations]

        # Calculate accuracy
        citation_accuracy = len(valid_citations) / len(found_citations) if found_citations else 1.0

        # Count unique citations
        unique_citations = len(set(found_citations))

        return {
            'citation_accuracy': citation_accuracy,
            'total_citations': len(found_citations),
            'unique_citations': unique_citations,
            'valid_citations': len(valid_citations),
            'invalid_citations': len(found_citations) - len(valid_citations)
        }

    def run_benchmark(self,
                     rag_pipeline,
                     test_queries: List[TestQuery],
                     k: int = 5,
                     use_generation: bool = True) -> EvaluationMetrics:
        """
        Run comprehensive benchmark on RAG system.

        Args:
            rag_pipeline: RAG pipeline instance
            test_queries: List of test queries with ground truth
            k: Number of results to retrieve
            use_generation: Whether to test generation

        Returns:
            Comprehensive evaluation metrics
        """
        retrieval_metrics_list = []
        retrieval_times = []
        generation_times = []
        citation_metrics_list = []

        print(f"Running benchmark with {len(test_queries)} test queries...")

        for i, test_query in enumerate(test_queries, 1):
            print(f"  Testing query {i}/{len(test_queries)}: {test_query.query[:50]}...")

            # Measure retrieval time
            start_time = time.time()
            rag_result = rag_pipeline.query(test_query.query, top_k=k)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)

            # Evaluate retrieval quality
            retrieved_chunks = [
                {'metadata': source['metadata']}
                for source in rag_result.sources
            ]
            retrieval_metrics = self.evaluate_retrieval(
                retrieved_chunks,
                test_query.relevant_doc_ids,
                k
            )
            retrieval_metrics_list.append(retrieval_metrics)

            # Test generation if enabled
            if use_generation and rag_pipeline.llm_generator:
                start_time = time.time()
                generated_response = rag_pipeline.query_with_generation(
                    test_query.query,
                    top_k=k
                )
                generation_time = time.time() - start_time
                generation_times.append(generation_time)

                # Evaluate citations
                available_citations = [f"[{i+1}]" for i in range(len(generated_response.citations))]
                citation_metrics = self.evaluate_citations(
                    generated_response.response,
                    available_citations
                )
                citation_metrics_list.append(citation_metrics)

        # Aggregate metrics
        avg_retrieval_metrics = {
            'precision_at_k': np.mean([m['precision_at_k'] for m in retrieval_metrics_list]),
            'recall_at_k': np.mean([m['recall_at_k'] for m in retrieval_metrics_list]),
            'mrr': np.mean([m['mrr'] for m in retrieval_metrics_list]),
            'ndcg_at_k': np.mean([m['ndcg_at_k'] for m in retrieval_metrics_list])
        }

        latency_metrics = self.evaluate_latency(
            retrieval_times,
            generation_times if generation_times else None
        )

        # Aggregate citation metrics
        avg_citation_accuracy = None
        avg_citations_per_response = None
        if citation_metrics_list:
            avg_citation_accuracy = np.mean([m['citation_accuracy'] for m in citation_metrics_list])
            avg_citations_per_response = np.mean([m['total_citations'] for m in citation_metrics_list])

        return EvaluationMetrics(
            precision_at_k=avg_retrieval_metrics['precision_at_k'],
            recall_at_k=avg_retrieval_metrics['recall_at_k'],
            mean_reciprocal_rank=avg_retrieval_metrics['mrr'],
            ndcg_at_k=avg_retrieval_metrics['ndcg_at_k'],
            avg_retrieval_time_ms=latency_metrics['avg_retrieval_time_ms'],
            avg_generation_time_ms=latency_metrics.get('avg_generation_time_ms'),
            avg_total_time_ms=latency_metrics.get('avg_total_time_ms', latency_metrics['avg_retrieval_time_ms']),
            citation_accuracy=avg_citation_accuracy,
            citations_per_response=avg_citations_per_response
        )

    def print_metrics(self, metrics: EvaluationMetrics):
        """
        Pretty print evaluation metrics.

        Args:
            metrics: Evaluation metrics to print
        """
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION RESULTS")
        print("="*60)

        print("\n📊 Retrieval Metrics:")
        print(f"  Precision@K:          {metrics.precision_at_k:.4f}")
        print(f"  Recall@K:             {metrics.recall_at_k:.4f}")
        print(f"  Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.4f}")
        print(f"  NDCG@K:               {metrics.ndcg_at_k:.4f}")

        print("\n⏱️  Latency Metrics:")
        print(f"  Avg Retrieval Time:   {metrics.avg_retrieval_time_ms:.2f} ms")
        if metrics.avg_generation_time_ms:
            print(f"  Avg Generation Time:  {metrics.avg_generation_time_ms:.2f} ms")
            print(f"  Avg Total Time:       {metrics.avg_total_time_ms:.2f} ms")

        if metrics.citation_accuracy is not None:
            print("\n📝 Citation Metrics:")
            print(f"  Citation Accuracy:    {metrics.citation_accuracy:.4f}")
            print(f"  Avg Citations/Response: {metrics.citations_per_response:.2f}")

        print("\n" + "="*60)


def create_test_queries() -> List[TestQuery]:
    """
    Create a set of test queries for evaluation.

    Returns:
        List of test queries
    """
    # These are example queries - should be customized based on actual papers
    test_queries = [
        TestQuery(
            query="What is the attention mechanism in transformers?",
            relevant_doc_ids=["1706.03762"],  # Attention is All You Need
            relevant_sections=["Introduction", "Model Architecture"]
        ),
        TestQuery(
            query="How does multi-head attention work?",
            relevant_doc_ids=["1706.03762"],
            relevant_sections=["Model Architecture"]
        ),
        TestQuery(
            query="What is positional encoding used for?",
            relevant_doc_ids=["1706.03762"],
            relevant_sections=["Model Architecture"]
        ),
        TestQuery(
            query="How are transformers trained?",
            relevant_doc_ids=["1706.03762"],
            relevant_sections=["Training"]
        ),
        TestQuery(
            query="What are the benefits of self-attention?",
            relevant_doc_ids=["1706.03762"],
            relevant_sections=["Introduction", "Model Architecture"]
        ),
    ]

    return test_queries


def main():
    """Test the evaluation framework."""
    # Create sample data
    evaluator = RAGEvaluator()

    # Test retrieval evaluation
    print("Testing retrieval evaluation...")
    sample_chunks = [
        {'metadata': {'arxiv_id': '1706.03762'}},
        {'metadata': {'arxiv_id': '1810.04805'}},
        {'metadata': {'arxiv_id': '1706.03762'}},
    ]
    relevant_ids = ['1706.03762']

    metrics = evaluator.evaluate_retrieval(sample_chunks, relevant_ids, k=3)
    print(f"  Precision: {metrics['precision_at_k']:.2f}")
    print(f"  Recall: {metrics['recall_at_k']:.2f}")
    print(f"  MRR: {metrics['mrr']:.2f}")
    print(f"  NDCG: {metrics['ndcg_at_k']:.2f}")

    # Test citation evaluation
    print("\nTesting citation evaluation...")
    response = "The attention mechanism [1] allows models to focus on relevant parts. Multi-head attention [2] improves this."
    available_citations = ["[1]", "[2]", "[3]"]

    citation_metrics = evaluator.evaluate_citations(response, available_citations)
    print(f"  Citation Accuracy: {citation_metrics['citation_accuracy']:.2f}")
    print(f"  Total Citations: {citation_metrics['total_citations']}")
    print(f"  Valid Citations: {citation_metrics['valid_citations']}")

    print("\n✅ Evaluation framework tests passed!")


if __name__ == "__main__":
    main()
