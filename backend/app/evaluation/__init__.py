"""Evaluation package exports."""

from backend.app.evaluation.benchmark_data import BENCHMARK_DATASET, BenchmarkTestCase
from backend.app.evaluation.evaluator import (
    BenchmarkResult,
    EvaluationReport,
    EvidenceMetricsReport,
    RetrievalMetricsReport,
    SecurityMetricsReport,
    SystemEvaluator,
    default_evaluator,
)
from backend.app.evaluation.metrics import (
    claim_support_rate,
    evidence_coverage,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reliability_metric,
    security_rejection_rate,
    source_diversity,
)

__all__ = [
    "BENCHMARK_DATASET",
    "BenchmarkTestCase",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "evidence_coverage",
    "claim_support_rate",
    "source_diversity",
    "reliability_metric",
    "security_rejection_rate",
    "SystemEvaluator",
    "default_evaluator",
    "EvaluationReport",
    "BenchmarkResult",
    "RetrievalMetricsReport",
    "EvidenceMetricsReport",
    "SecurityMetricsReport",
]
