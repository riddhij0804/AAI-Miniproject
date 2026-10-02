"""Credibility Assessment Agent evaluating sources with transparent, inspectable factor scoring."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

from backend.app.schemas.common import SourceType, generate_uuid, utc_now
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class CredibilityAssessmentAgent:
    """Evaluates the credibility of research sources using multi-factor transparent criteria.

    Scores range from 0.0 to 1.0. All scores include human-inspectable reasoning.
    """

    # Domain reputation priors
    HIGH_AUTHORITY_DOMAINS = {
        "nature.com": 0.95,
        "science.org": 0.95,
        "thelancet.com": 0.95,
        "nejm.org": 0.95,
        "cell.com": 0.93,
        "pnas.org": 0.92,
        "arxiv.org": 0.85,
        "biorxiv.org": 0.83,
        "medrxiv.org": 0.84,
        "nih.gov": 0.94,
        "cdc.gov": 0.92,
        "who.int": 0.92,
        "ieee.org": 0.90,
        "acm.org": 0.90,
        "sciencedirect.com": 0.88,
        "springer.com": 0.88,
        "reuters.com": 0.88,
        "apnews.com": 0.88,
        "bbc.com": 0.85,
        "bbc.co.uk": 0.85,
        "thehindu.com": 0.82,
        "nytimes.com": 0.85,
        "wsj.com": 0.85,
    }

    # Factor weights for overall composite score
    WEIGHTS = {
        "authority": 0.25,
        "primary_source": 0.20,
        "recency": 0.15,
        "evidence_quality": 0.20,
        "corroboration": 0.20,
    }

    def assess_source(
        self,
        source: Source,
        all_sources: Optional[List[Source]] = None,
    ) -> CredibilityAssessment:
        """Calculate transparent credibility factor scores and generate explanatory reasoning."""
        parsed_url = urlparse(source.url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        reasons: List[str] = []

        # 1. Authority Score
        authority_score, authority_reason = self._compute_authority(domain, source.source_type)
        reasons.append(f"Authority ({authority_score:.2f}): {authority_reason}")

        # 2. Primary Source Score
        primary_score, primary_reason = self._compute_primary_source(source)
        reasons.append(f"Primary Source ({primary_score:.2f}): {primary_reason}")

        # 3. Recency Score
        recency_score, recency_reason = self._compute_recency(source.published_at)
        reasons.append(f"Recency ({recency_score:.2f}): {recency_reason}")

        # 4. Evidence Quality Score
        quality_score, quality_reason = self._compute_evidence_quality(source)
        reasons.append(f"Evidence Quality ({quality_score:.2f}): {quality_reason}")

        # 5. Peer Review Score (where applicable)
        peer_review_score, peer_review_reason = self._compute_peer_review(source, domain)
        if peer_review_score is not None:
            reasons.append(f"Peer Review ({peer_review_score:.2f}): {peer_review_reason}")

        # 6. Corroboration Score (agreement across collection)
        corroboration_score, corroboration_reason = self._compute_corroboration(source, all_sources or [source])
        reasons.append(f"Corroboration ({corroboration_score:.2f}): {corroboration_reason}")

        # 7. Composite Weighted Score
        overall_score = (
            self.WEIGHTS["authority"] * authority_score
            + self.WEIGHTS["primary_source"] * primary_score
            + self.WEIGHTS["recency"] * recency_score
            + self.WEIGHTS["evidence_quality"] * quality_score
            + self.WEIGHTS["corroboration"] * corroboration_score
        )

        # Slight boost if formally peer-reviewed
        if peer_review_score and peer_review_score >= 0.8:
            overall_score = min(1.0, overall_score + 0.05)

        overall_score = round(overall_score, 2)
        detailed_reasoning = "\n".join(reasons)

        return CredibilityAssessment(
            id=generate_uuid(),
            source_id=source.id,
            authority_score=round(authority_score, 2),
            primary_source_score=round(primary_score, 2),
            recency_score=round(recency_score, 2),
            evidence_quality_score=round(quality_score, 2),
            peer_review_score=round(peer_review_score, 2) if peer_review_score is not None else None,
            corroboration_score=round(corroboration_score, 2),
            overall_score=overall_score,
            reasoning=detailed_reasoning,
            metadata={
                "domain": domain,
                "factor_weights": self.WEIGHTS,
                "assessed_at": utc_now().isoformat(),
            },
        )

    def _compute_authority(self, domain: str, source_type: SourceType) -> tuple[float, str]:
        # Exact known domain check
        for known, score in self.HIGH_AUTHORITY_DOMAINS.items():
            if domain == known or domain.endswith("." + known):
                return score, f"Recognized high-impact institution/publisher '{known}'"

        if domain.endswith(".edu") or ".edu." in domain or "ac.uk" in domain:
            return 0.88, "Accredited educational/university domain"
        if domain.endswith(".gov") or ".gov." in domain:
            return 0.90, "Official government agency domain"
        if source_type == SourceType.ACADEMIC_JOURNAL:
            return 0.85, "Academic or scholarly journal repository"
        if source_type == SourceType.GOVERNMENT:
            return 0.88, "Government publication"
        if source_type == SourceType.NEWS:
            return 0.75, "Mainstream news publication"
        if domain.endswith(".org"):
            return 0.70, "Non-profit or non-governmental organization domain"

        return 0.55, f"Standard commercial or personal web domain '{domain}'"

    def _compute_primary_source(self, source: Source) -> tuple[float, str]:
        if source.source_type in (SourceType.ACADEMIC_JOURNAL, SourceType.PREPRINT):
            return 0.95, "Original academic research publication"
        if source.source_type == SourceType.GOVERNMENT:
            return 0.90, "Primary government or regulatory data release"
        if source.source_type == SourceType.TECHNICAL_REPORT:
            return 0.80, "Technical research report or whitepaper"
        if source.source_type == SourceType.NEWS:
            return 0.65, "Secondary news report; may summarize primary research"
        return 0.50, "General webpage or secondary commentary"

    def _compute_recency(self, published_at: Optional[datetime]) -> tuple[float, str]:
        if not published_at:
            return 0.65, "Publication date not specified; assigned baseline recency"

        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            pub = published_at.replace(tzinfo=timezone.utc)
        else:
            pub = published_at

        days_old = max(0, (now - pub).days)
        years_old = days_old / 365.25

        if years_old <= 1.0:
            return 1.0, f"Published within the last year ({days_old} days ago)"
        elif years_old <= 3.0:
            return 0.90, f"Published within 3 years ({years_old:.1f} years ago)"
        elif years_old <= 5.0:
            return 0.80, f"Published within 5 years ({years_old:.1f} years ago)"
        elif years_old <= 10.0:
            return 0.65, f"Published between 5-10 years ago ({years_old:.1f} years ago)"
        else:
            return 0.50, f"Historical document published >10 years ago ({years_old:.1f} years ago)"

    def _compute_evidence_quality(self, source: Source) -> tuple[float, str]:
        score = 0.60
        reasons = []

        if source.author:
            score += 0.10
            reasons.append("attributed byline/author")
        if source.publisher:
            score += 0.10
            reasons.append(f"identified publisher '{source.publisher}'")
        if source.source_type in (SourceType.ACADEMIC_JOURNAL, SourceType.GOVERNMENT):
            score += 0.15
            reasons.append("rigorous publishing standards")

        return min(1.0, score), f"Quality indicators: {', '.join(reasons) if reasons else 'baseline formatting'}"

    def _compute_peer_review(self, source: Source, domain: str) -> tuple[Optional[float], str]:
        if source.source_type == SourceType.ACADEMIC_JOURNAL:
            return 0.95, "Published in formal peer-reviewed scholarly journal"
        if source.source_type == SourceType.PREPRINT:
            return 0.40, "Preprint manuscript (un-peer-reviewed scholarly draft)"
        if any(j in domain for j in ["nature.com", "science.org", "thelancet.com", "cell.com", "pnas.org"]):
            return 0.95, f"Indexed in premier peer-reviewed journal ({domain})"
        return None, "Peer-review metric not applicable for non-academic source"

    def _compute_corroboration(self, source: Source, all_sources: List[Source]) -> tuple[float, str]:
        if len(all_sources) <= 1:
            return 0.70, "Single source evaluated; neutral corroboration baseline"

        # Check domain diversity across sources
        other_domains = {
            urlparse(s.url).netloc.lower()
            for s in all_sources
            if s.id != source.id and s.url
        }

        if len(other_domains) >= 3:
            return 0.90, f"Multiple independent publishers ({len(other_domains)} distinct domains) present in session"
        elif len(other_domains) >= 1:
            return 0.80, f"Corroborated by {len(other_domains)} additional publisher(s)"
        return 0.65, "Limited publisher diversity in current batch"
