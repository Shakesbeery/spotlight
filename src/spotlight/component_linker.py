"""
Component Linker and Entity Resolution Module.
1. Associates mechanical failures with specific device components (e.g. sheath hub, tulip collar, balloon).
2. Reconciles true duplicates (acronym coreference, parenthetical mentions) while preserving
   distinct manifestations under ontology collapse (e.g. failure to fire vs failure to cut under A0502).
"""

from typing import List, Optional, Tuple, Dict
import re
from spotlight.schemas import ExtractedFinding, CategoryType


# Canonical list of physical device components and sub-assemblies (IMDRF Annex G / GHTF device anatomy)
DEVICE_COMPONENTS = [
    r"\b(?:polyaxial\s+)?tulip\s+(?:collar|head)\b",
    r"\b(?:introducer\s+)?sheath\s+hub\b",
    r"\bhemostasis\s+valve(?:\s+junction)?\b",
    r"\b(?:delivery\s+)?balloon\b",
    r"\b(?:crimped\s+|coronary\s+)?stent\b",
    r"\b(?:distal\s+)?cannula\b",
    r"\b(?:stapler\s+)?(?:jaw|jaws|anvil)\b",
    r"\b(?:blade|knife|cutter)\b",
    r"\b(?:guidewire|delivery\s+wire|wire)\b",
    r"\b(?:powered\s+)?(?:stapler\s+)?(?:handle|motor\s+drive)\b",
    r"\b(?:catheter\s+)?(?:shaft|tip|lumen)\b",
    r"\bpedicle\s+screw(?:\s+shank)?\b",
    r"\bset\s+screw\b",
    r"\b(?:femoral\s+)?(?:implant|component|shell)\b",
    r"\bsterile\s+(?:barrier\s+)?(?:tray|pouch|blister\s+pack(?:aging)?)\b",
    r"\bpackaging\s+seal\b",
    r"\bsealing\s+cap\b",
    r"\b(?:pacing\s+)?lead\b",
]


class ComponentLinker:
    """
    Extracts and links physical device components to failure modes,
    and resolves coreference/acronym duplicates.
    """

    def __init__(self):
        self.compiled_components = [re.compile(p, re.IGNORECASE) for p in DEVICE_COMPONENTS]

    def extract_component(self, text: str, span_start: int, span_end: int) -> Optional[str]:
        """
        Searches the clause/sentence containing [span_start, span_end] for the most
        syntactically proximate device component.
        """
        # Expand search window to 60 characters before and after the failure span
        window_start = max(0, span_start - 60)
        window_end = min(len(text), span_end + 60)
        window_text = text[window_start:window_end]

        best_component = None
        min_dist = float("inf")

        for comp_pat in self.compiled_components:
            for match in comp_pat.finditer(window_text):
                comp_span = match.group(0)
                comp_center = window_start + (match.start() + match.end()) / 2
                span_center = (span_start + span_end) / 2
                dist = abs(comp_center - span_center)

                if dist < min_dist:
                    min_dist = dist
                    best_component = comp_span.strip()

        return best_component

    def link_components_to_findings(
        self, narrative: str, findings: List[ExtractedFinding]
    ) -> List[ExtractedFinding]:
        """Enriches findings with linked device components where applicable."""
        enriched = []
        for f in findings:
            comp = f.affected_component
            mode = f.failure_mode

            if f.category in (CategoryType.OPERATIONAL_PROBLEM, CategoryType.MANUFACTURING_ISSUE):
                if f.start_char is not None and f.end_char is not None:
                    detected_comp = self.extract_component(narrative, f.start_char, f.end_char)
                    if detected_comp:
                        comp = detected_comp

                # Infer failure mode from verbatim span if not provided
                if not mode:
                    mode = f.verbatim_span

            enriched.append(
                ExtractedFinding(
                    id=f.id,
                    category=f.category,
                    verbatim_span=f.verbatim_span,
                    start_char=f.start_char,
                    end_char=f.end_char,
                    temporal_timing=f.temporal_timing,
                    confidence=f.confidence,
                    affected_component=comp,
                    failure_mode=mode,
                    aliases=f.aliases,
                    triggered_by=f.triggered_by,
                    normalized_terms=f.normalized_terms,
                    details=f.details,
                )
            )
        return enriched


class EntityResolver:
    """
    Reconciles True Duplicates (e.g. 'coronary artery bypass graft' vs 'CABG')
    while preserving multi-aspect manifestations under ontology collapse (e.g. 'misfire' vs 'failure to cut').
    """

    @staticmethod
    def _is_acronym(full_phrase: str, candidate_acronym: str) -> bool:
        """Checks if candidate_acronym is the acronym of full_phrase (e.g. CABG for coronary artery bypass graft)."""
        clean_full = re.sub(r"[^a-zA-Z\s]", "", full_phrase).strip()
        words = clean_full.split()
        if len(words) < 2:
            return False
        first_letters = "".join(w[0] for w in words).upper()
        cand = candidate_acronym.strip().upper()
        return cand == first_letters or cand in first_letters or first_letters in cand

    @classmethod
    def reconcile_duplicates(
        cls, findings: List[ExtractedFinding], narrative: str
    ) -> List[ExtractedFinding]:
        """
        Consolidates coreferent mentions and acronyms into single findings with aliases.
        Preserves distinct physical events even if they map to identical ontology codes.
        """
        if len(findings) <= 1:
            return findings

        consolidated: List[ExtractedFinding] = []
        skip_indices = set()

        for i, f1 in enumerate(findings):
            if i in skip_indices:
                continue

            current_finding = f1
            collected_aliases = list(f1.aliases)

            for j in range(i + 1, len(findings)):
                if j in skip_indices:
                    continue

                f2 = findings[j]

                # Condition 1: Acronym duplicate within the same category
                # e.g. "coronary artery bypass graft" and "CABG"
                is_acronym_pair = (
                    f1.category == f2.category
                    and (cls._is_acronym(f1.verbatim_span, f2.verbatim_span)
                         or cls._is_acronym(f2.verbatim_span, f1.verbatim_span))
                )

                # Condition 2: Immediate parenthetical mention: "X (Y)"
                span1, span2 = f1.verbatim_span.lower(), f2.verbatim_span.lower()
                is_parenthetical = (
                    f1.category == f2.category
                    and (f"{span1} ({span2})" in narrative.lower() or f"{span2} ({span1})" in narrative.lower())
                )

                # Condition 3: Exact same verbatim span or complete substring containment of same concept
                is_exact_synonym = (
                    f1.category == f2.category
                    and (span1 == span2 or (len(span1) > len(span2) and span2 in span1 and len(span2) > 4))
                    and f1.affected_component == f2.affected_component
                )

                if is_acronym_pair or is_parenthetical or is_exact_synonym:
                    # TRUE DUPLICATE DETECTED: Merge into current_finding and add alias
                    skip_indices.add(j)
                    alias_to_add = f2.verbatim_span if f2.verbatim_span != current_finding.verbatim_span else None
                    if alias_to_add and alias_to_add not in collected_aliases:
                        collected_aliases.append(alias_to_add)

                # NOTE: If f1 and f2 share the same ontology code (e.g. A0502), but are NOT acronyms
                # and describe different physical actions (e.g. "misfired" vs "failure to cut"),
                # they DO NOT match the above conditions and are PRESERVED as distinct findings!

            # Update finding with resolved aliases
            merged_finding = ExtractedFinding(
                id=current_finding.id,
                category=current_finding.category,
                verbatim_span=current_finding.verbatim_span,
                start_char=current_finding.start_char,
                end_char=current_finding.end_char,
                temporal_timing=current_finding.temporal_timing,
                confidence=current_finding.confidence,
                affected_component=current_finding.affected_component,
                failure_mode=current_finding.failure_mode,
                aliases=collected_aliases,
                triggered_by=current_finding.triggered_by,
                normalized_terms=current_finding.normalized_terms,
                details=current_finding.details,
            )
            consolidated.append(merged_finding)

        return consolidated
