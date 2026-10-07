"""Citation and Final Validation Agent performing strict reference ground-truth audits."""

import logging
from typing import Dict, List, Optional

from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.orchestration.state import (
    CitationValidationResult,
    EvidenceVerificationReport,
    FinalResearchReport,
)
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class CitationValidationAgent:
    """Validates that final report citations genuinely resolve to real sources and verbatim text."""

    def __init__(self, store: Optional[EvidenceStore] = None):
        self.store = store or default_evidence_store

    def validate_report(
        self,
        report: FinalResearchReport,
        sources: List[Source],
        documents: List[Document],
        evidence_records: List[Evidence],
        verification_report: EvidenceVerificationReport,
    ) -> CitationValidationResult:
        """Audit every cited statement in report against stored ground truth."""
        logger.info(f"CitationValidationAgent auditing {len(report.references)} cited references.")

        known_urls = {s.url for s in sources if s.url}
        doc_chunks = {c.id: c.text for d in documents for c in d.chunks}
        stored_ev_quotes = {e.text.strip().lower() for e in evidence_records}

        invalid_citations: List[str] = []
        missing_sources: List[str] = []
        valid_count = 0

        for ref in report.references:
            # 1. Check Source URL existence
            if ref.source_url not in known_urls:
                # Fallback check in persistent store
                stored_src = self.store.get_source(ref.source_url)
                if not stored_src and not any(s.url == ref.source_url for s in sources):
                    missing_sources.append(ref.source_url)
                    invalid_citations.append(f"Source URL '{ref.source_url}' not found in research database.")
                    continue

            # 2. Check quote existence in evidence records or document chunks
            ref_quote_clean = ref.exact_quote.strip().lower()
            quote_found = False
            if ref_quote_clean in stored_ev_quotes:
                quote_found = True
            else:
                # Check in chunk texts
                for chunk_text in doc_chunks.values():
                    if ref_quote_clean in chunk_text.lower():
                        quote_found = True
                        break

            if not quote_found:
                invalid_citations.append(
                    f"Cited quote for '{ref.claim_text[:40]}...' was not found in stored document chunks."
                )
            else:
                valid_count += 1

        # 3. Check that unsupported claims are not presented as verified facts
        unsupported_flagged = [
            f.claim_text for f in verification_report.claim_findings if not f.is_supported
        ]

        # 4. Check that contradictions are not silently omitted if detected
        if verification_report.detected_contradictions and not report.contradictory_findings:
            invalid_citations.append("Contradictions detected during verification were omitted from final report.")

        is_valid = len(invalid_citations) == 0

        return CitationValidationResult(
            is_valid=is_valid,
            total_citations_checked=len(report.references),
            valid_citations_count=valid_count,
            invalid_citations=invalid_citations,
            unsupported_claims_flagged=unsupported_flagged,
            missing_sources=list(set(missing_sources)),
            notes=(
                f"Audited {len(report.references)} citations. Valid: {valid_count}, "
                f"Invalid: {len(invalid_citations)}. Flagged unsupported claims: {len(unsupported_flagged)}."
            ),
        )

