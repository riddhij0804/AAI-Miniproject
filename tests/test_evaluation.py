"""Unit tests for Evaluation module and metrics."""

import pytest
from backend.app.evaluation.benchmark_data import BENCHMARK_DATASET
from backend.app.evaluation.evaluator import SystemEvaluator
from backend.app.evaluation.metrics import (
    claim_support_rate,
    evidence_coverage,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    security_rejection_rate,
    source_diversity,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


def test_retrieval_metrics_calculation():
    ground_truth = {"c1", "c2", "c3"}
    retrieved = ["c1", "x1", "c2", "x2", "x3"]

    assert recall_at_k(retrieved, ground_truth, k=3) == round(2 / 3, 4)
    assert recall_at_k(retrieved, ground_truth, k=5) == round(2 / 3, 4)
    assert precision_at_k(retrieved, ground_truth, k=3) == round(2 / 3, 4)
    assert mrr(retrieved, ground_truth) == 1.0  # First item "c1" is in ground truth
    assert ndcg_at_k(retrieved, ground_truth, k=3) > 0.5


def test_evidence_coverage_and_support_rate():
    claims = [
        Claim(text="Agricultural overuse of antibiotics accelerates selection pressure."),
        Claim(text="Hospital transmission spreads resistant plasmids."),
    ]
    topics = ["agricultural overuse", "hospital transmission", "diagnostics"]
    cov = evidence_coverage(claims, topics)
    assert cov == round(2 / 3, 4)

    ev_records = [
        Evidence(
            source_id="s1",
            document_id="d1",
            chunk_id="c1",
            text="Quote 1",
            location="Loc 1",
            metadata={"grounding_verified": True},
        ),
        Evidence(
            source_id="s2",
            document_id="d2",
            chunk_id="c2",
            text="Quote 2",
            location="Loc 2",
            metadata={"grounding_verified": False},
        ),
    ]
    assert claim_support_rate(ev_records) == 0.5


def test_source_diversity_calculation():
    sources = [
        Source(url="https://who.int/amr", title="WHO"),
        Source(url="https://cdc.gov/amr", title="CDC"),
        Source(url="https://who.int/other", title="WHO 2"),
    ]
    # Unique domains: who.int, cdc.gov (2 unique out of 3)
    assert source_diversity(sources) == round(2 / 3, 4)


@pytest.mark.asyncio
async def test_system_evaluator_runs_benchmark():
    evaluator = SystemEvaluator()
    # Run on first test case from benchmark dataset
    test_subset = [BENCHMARK_DATASET[0]]
    report = await evaluator.run_benchmark(test_subset)

    assert report.total_test_cases == 1
    assert report.mean_recall_at_3 > 0.0
    assert report.mean_precision_at_3 > 0.0
    assert report.mean_mrr > 0.0
    assert report.overall_security_rejection_rate == 1.0
    assert len(report.case_results) == 1
