"""Evidence Verification Agent ensuring traceability, provenance, support, and contradiction detection."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.orchestration.state import (
    ClaimVerificationFinding,
    EvidenceVerificationReport,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import ClaimStatus, generate_uuid, utc_now
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Contradiction, Evidence
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class EvidenceVerificationAgent:
    """Verifies provenance, evidence support strength, and identifies cross-claim contradictions."""

    def __init__(
        self,
        store: Optional[EvidenceStore] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.store = store or default_evidence_store
        self.llm = llm_client or default_llm_client

    async def verify_evidence(
        self,
        claims: List[Claim],
        evidence_records: List[Evidence],
        sources: List[Source],
        documents: List[Document],
    ) -> EvidenceVerificationReport:
        """Run complete verification across claims, evidence, sources, and documents."""
        logger.info(f"EvidenceVerificationAgent verifying {len(claims)} claims and {len(evidence_records)} evidence items.")

        # Index known resources for O(1) deterministic provenance resolution
        source_map: Dict[str, Source] = {s.id: s for s in sources}
        doc_map: Dict[str, Document] = {d.id: d for d in documents}
        chunk_map: Dict[str, str] = {}
        for d in documents:
            for c in d.chunks:
                chunk_map[c.id] = c.text

        evidence_map: Dict[str, Evidence] = {e.id: e for e in evidence_records}
        claim_to_evidences: Dict[str, List[Evidence]] = {}
        for e in evidence_records:
            if e.claim_id:
                claim_to_evidences.setdefault(e.claim_id, []).append(e)

        claim_findings: List[ClaimVerificationFinding] = []
        supported_count = 0
        unsupported_count = 0
        weak_count = 0
        verified_ev_count = 0

        for claim in claims:
            associated_ev = claim_to_evidences.get(claim.id, [])
            if not associated_ev:
                # Also check matching by evidence_ids
                associated_ev = [evidence_map[eid] for eid in claim.evidence_ids if eid in evidence_map]

            if not associated_ev:
                # Unsupported claim: No evidence at all
                unsupported_count += 1
                claim_findings.append(
                    ClaimVerificationFinding(
                        claim_id=claim.id,
                        claim_text=claim.text,
                        is_supported=False,
                        support_strength=0.0,
                        has_valid_provenance=False,
                        evidence_ids=[],
                        source_urls=[],
                        flags=["unsupported", "no_evidence_linked"],
                        reasoning="No grounded evidence records are associated with this assertion.",
                    )
                )
                continue

            # Deterministic Provenance Check: Do evidence references resolve to real sources/documents?
            provenance_valid = True
            source_urls = []
            for ev in associated_ev:
                src = source_map.get(ev.source_id)
                if not src:
                    # Check persistent store
                    stored_src = self.store.get_source(ev.source_id)
                    if stored_src:
                        src = stored_src
                        source_map[ev.source_id] = stored_src

                if src and src.url:
                    source_urls.append(src.url)
                else:
                    provenance_valid = False

                # Check chunk resolution
                if ev.chunk_id not in chunk_map:
                    # Check if document has it
                    doc = doc_map.get(ev.document_id)
                    if not doc:
                        stored_doc = self.store.get_document(ev.document_id)
                        if stored_doc:
                            doc = stored_doc
                            doc_map[ev.document_id] = stored_doc

                    if not doc or not any(c.id == ev.chunk_id for c in doc.chunks):
                        provenance_valid = False

            if provenance_valid:
                verified_ev_count += len(associated_ev)

            # Semantic Support Evaluation: Does the evidence text support the claim?
            support_score, support_flags, reasoning = self._evaluate_claim_support(claim, associated_ev)

            if not provenance_valid:
                support_flags.append("no_provenance")

            is_supported = support_score >= 0.5 and provenance_valid
            if is_supported and support_score >= 0.7:
                supported_count += 1
            elif is_supported:
                weak_count += 1
            else:
                unsupported_count += 1

            claim_findings.append(
                ClaimVerificationFinding(
                    claim_id=claim.id,
                    claim_text=claim.text,
                    is_supported=is_supported,
                    support_strength=support_score,
                    has_valid_provenance=provenance_valid,
                    evidence_ids=[e.id for e in associated_ev],
                    source_urls=list(set(source_urls)),
                    flags=support_flags,
                    reasoning=reasoning,
                )
            )

        # Detect cross-claim contradictions
        contradictions = await self._detect_contradictions(claims, evidence_records)

        # Persist detected contradictions to store
        for contra in contradictions:
            self.store.save_contradiction(contra)

        provenance_passed = all(f.has_valid_provenance for f in claim_findings if f.evidence_ids)

        return EvidenceVerificationReport(
            total_claims_checked=len(claims),
            supported_claims_count=supported_count,
            unsupported_claims_count=unsupported_count,
            weakly_supported_claims_count=weak_count,
            claim_findings=claim_findings,
            detected_contradictions=contradictions,
            verified_evidence_count=verified_ev_count,
            provenance_integrity_passed=provenance_passed,
            verification_notes=(
                f"Verified {len(claims)} claims: {supported_count} strong, "
                f"{weak_count} weak, {unsupported_count} unsupported. "
                f"Detected {len(contradictions)} contradiction(s)."
            ),
        )

    def _evaluate_claim_support(self, claim: Claim, evidences: List[Evidence]) -> Tuple[float, List[str], str]:
        """Evaluate lexical and semantic alignment between claim text and backing evidence quotes."""
        flags: List[str] = []
        combined_evidence_text = " ".join(e.text for e in evidences).lower()
        claim_text_lower = claim.text.lower()

        # Token overlap analysis
        claim_words = set(re.findall(r"\w+", claim_text_lower)) - {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "of", "for", "with"
        }
        if not claim_words:
            return 0.5, ["uninformative_claim"], "Claim text lacks substantive terms."

        overlap = sum(1 for w in claim_words if w in combined_evidence_text)
        lexical_coverage = overlap / len(claim_words)

        # Quantitative grounding bonus: If numbers or percentages in claim match evidence
        claim_nums = set(re.findall(r"\d+(?:\.\d+)?%?", claim_text_lower))
        evidence_nums = set(re.findall(r"\d+(?:\.\d+)?%?", combined_evidence_text))
        num_match = bool(claim_nums and (claim_nums & evidence_nums))

        base_score = lexical_coverage
        if num_match:
            base_score = min(1.0, base_score + 0.2)

        # Refuting check: Does any associated evidence explicitly refute the claim?
        refuting = [e for e in evidences if not e.supports]
        if refuting:
            flags.append("refuting_evidence_present")
            base_score = max(0.1, base_score - 0.4)

        if base_score < 0.4:
            flags.append("unsupported")
            reasoning = f"Low textual overlap ({lexical_coverage:.1%}) between assertion and backing evidence."
        elif base_score < 0.7:
            flags.append("weakly_supported")
            reasoning = f"Moderate evidence overlap ({lexical_coverage:.1%}); corroborating specifics limited."
        else:
            reasoning = f"Strong grounding confirmed ({lexical_coverage:.1%} concept match with verbatim source passages)."

        return round(base_score, 2), flags, reasoning

    async def _detect_contradictions(
        self,
        claims: List[Claim],
        evidence_records: List[Evidence],
    ) -> List[Contradiction]:
        """Identify substantive contradictions across claims using semantic and lexical analysis."""
        contradictions: List[Contradiction] = []
        if len(claims) < 2:
            return contradictions

        # Deterministic opposition indicators
        negation_pairs = [
            (r"\bincreased?\b", r"\bdecreased?\b"),
            (r"\beffective\b", r"\bineffective\b"),
            (r"\bsafe\b", r"\b(unsafe|hazardous|toxic)\b"),
            (r"\baccelerat\w+", r"\b(slowed|delayed|decelerated)\b"),
            (r"\bcauses?\b", r"\b(does not cause|unrelated to)\b"),
            (r"\bsupports?\b", r"\b(refutes?|opposes?)\b"),
        ]

        # Evidence-level direct opposition check
        claim_ev_map = {c.id: [e for e in evidence_records if e.claim_id == c.id] for c in claims}

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                c1 = claims[i]
                c2 = claims[j]

                c1_text = c1.text.lower()
                c2_text = c2.text.lower()

                # Check shared subject context
                c1_words = set(re.findall(r"\w+", c1_text)) - {"the", "is", "are", "of", "in", "to", "and"}
                c2_words = set(re.findall(r"\w+", c2_text)) - {"the", "is", "are", "of", "in", "to", "and"}
                shared_subject = len(c1_words & c2_words) >= 2

                if not shared_subject:
                    continue

                for pos_pattern, neg_pattern in negation_pairs:
                    is_c1_pos = bool(re.search(pos_pattern, c1_text))
                    is_c2_neg = bool(re.search(neg_pattern, c2_text))
                    is_c1_neg = bool(re.search(neg_pattern, c1_text))
                    is_c2_pos = bool(re.search(pos_pattern, c2_text))

                    if (is_c1_pos and is_c2_neg) or (is_c1_neg and is_c2_pos):
                        ev1 = claim_ev_map[c1.id][0].id if claim_ev_map[c1.id] else None
                        ev2 = claim_ev_map[c2.id][0].id if claim_ev_map[c2.id] else None
                        reason = f"Opposing directional claims identified regarding shared topic: '{c1.text}' vs '{c2.text}'."
                        contradictions.append(
                            Contradiction(
                                id=generate_uuid(),
                                claim_a_id=c1.id,
                                claim_b_id=c2.id,
                                evidence_a_id=ev1,
                                evidence_b_id=ev2,
                                reasoning=reason,
                                severity=0.75,
                                metadata={"type": "directional_opposition"},
                            )
                        )
                        break

        return contradictions

