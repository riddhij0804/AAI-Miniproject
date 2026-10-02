"""Unit tests for Credibility Assessment Agent."""

from datetime import datetime, timezone, timedelta
from backend.app.agents.credibility import CredibilityAssessmentAgent
from backend.app.schemas.common import SourceType
from backend.app.schemas.source import Source


def test_credibility_academic_journal():
    agent = CredibilityAssessmentAgent()
    now = datetime.now(timezone.utc)
    source = Source(
        id="src-lancet-1",
        url="https://www.thelancet.com/journals/lancet/article/AMR2024",
        title="Global Antimicrobial Resistance Surveillance",
        publisher="The Lancet",
        author="Murray et al.",
        source_type=SourceType.ACADEMIC_JOURNAL,
        published_at=now - timedelta(days=60),  # 2 months old
    )

    assessment = agent.assess_source(source)
    assert assessment.authority_score >= 0.85
    assert assessment.primary_source_score >= 0.90
    assert assessment.recency_score == 1.0  # Under 1 year old
    assert assessment.peer_review_score is not None
    assert assessment.peer_review_score >= 0.90
    assert assessment.overall_score >= 0.85
    # Verify reasoning is present and informative
    assert "The Lancet" in assessment.reasoning or "scholarly" in assessment.reasoning
    assert "Peer Review" in assessment.reasoning


def test_credibility_unverified_blog():
    agent = CredibilityAssessmentAgent()
    source = Source(
        id="src-blog-1",
        url="https://random-opinion-blog.net/amr-thoughts",
        title="My Thoughts on Antibiotics",
        publisher=None,
        author=None,
        source_type=SourceType.WEBPAGE,
        published_at=None,
    )

    assessment = agent.assess_source(source)
    assert assessment.authority_score <= 0.60
    assert assessment.primary_source_score <= 0.60
    assert assessment.peer_review_score is None
    assert assessment.overall_score < 0.70
    assert "random-opinion-blog.net" in assessment.reasoning


def test_credibility_corroboration_across_sources():
    agent = CredibilityAssessmentAgent()
    s1 = Source(id="s1", url="https://who.int/doc1", title="WHO", source_type=SourceType.GOVERNMENT)
    s2 = Source(id="s2", url="https://cdc.gov/doc2", title="CDC", source_type=SourceType.GOVERNMENT)
    s3 = Source(id="s3", url="https://nature.com/doc3", title="Nature", source_type=SourceType.ACADEMIC_JOURNAL)

    assessment_with_peers = agent.assess_source(s1, all_sources=[s1, s2, s3])
    assessment_solo = agent.assess_source(s1, all_sources=[s1])

    assert assessment_with_peers.corroboration_score > assessment_solo.corroboration_score
