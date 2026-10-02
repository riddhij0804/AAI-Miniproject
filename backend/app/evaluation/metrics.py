"""Evaluation metrics for retrieval, evidence grounding, reliability, and security."""

import math
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from backend.app.schemas.claim import Claim
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top-k."""
    if not ground_truth_ids:
        return 1.0
    top_k_ids = set(retrieved_ids[:k])
    relevant_retrieved = top_k_ids.intersection(ground_truth_ids)
    return round(len(relevant_retrieved) / len(ground_truth_ids), 4)


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Precision@K: proportion of retrieved items in top-k that are relevant."""
    if k <= 0:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0
    hits = sum(1 for cid in top_k_ids if cid in ground_truth_ids)
    return round(hits / len(top_k_ids), 4)


def mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for the first relevant item."""
    if not ground_truth_ids:
        return 1.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in ground_truth_ids:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ground_truth_ids), k) + 1))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def evidence_coverage(claims: List[Claim], expected_topics: List[str]) -> float:
    """Calculate evidence coverage: fraction of key expected concepts present in extracted claims."""
    if not expected_topics:
        return 1.0
    all_text = " ".join(c.text.lower() for c in claims)
    covered = sum(1 for topic in expected_topics if topic.lower() in all_text)
    return round(covered / len(expected_topics), 4)


def claim_support_rate(evidence_records: List[Evidence]) -> float:
    """Calculate claim support rate: fraction of evidence records backed by verified document grounding."""
    if not evidence_records:
        return 0.0
    grounded = sum(1 for e in evidence_records if e.metadata.get("grounding_verified", False))
    return round(grounded / len(evidence_records), 4)


def source_diversity(sources: List[Source]) -> float:
    """Calculate source diversity as unique publisher domain ratio [0.0 - 1.0]."""
    if not sources:
        return 0.0
    domains = set()
    for s in sources:
        try:
            dom = urlparse(s.url).netloc.lower()
            if dom.startswith("www."):
                dom = dom[4:]
            if dom:
                domains.add(dom)
        except Exception:
            pass
    return round(len(domains) / len(sources), 4)


def reliability_metric(successful_operations: int, total_operations: int) -> float:
    """Calculate operational success / reliability rate."""
    if total_operations <= 0:
        return 1.0
    return round(successful_operations / total_operations, 4)


def security_rejection_rate(blocked_attacks: int, total_attacks: int) -> float:
    """Calculate the percentage of adversarial or malicious inputs successfully mitigated."""
    if total_attacks <= 0:
        return 1.0
    return round(blocked_attacks / total_attacks, 4)
