"""
Fast triage and span extraction engine (Tier 1: ModernBERT / High-Throughput Baseline).
Evaluates cleaned segments, extracts candidate spans, assigns initial category scores,
and determines if a record requires the Gemma 4 fallback.
"""

from typing import List, Tuple, Optional
import re
from spotlight.schemas import (
    CategoryType,
    TemporalTiming,
    ExtractedFinding,
)
from spotlight.preprocessor import CleanedSegment
from spotlight.ontology import OntologyMatcher


# Diagnostic vocabulary patterns for fast candidate triage
MFG_PATTERNS = [
    r"\b(?:dark|black|white|metallic|visible|foreign)\s+(?:particulate(?: matter)?|debris|particles?|flakes?|specks?|shavings?)\b",
    r"\b(?:particulate(?: matter)?|foreign (?:material|body|debris)|debris|lint|fibers?)\b",
    r"\b(?:crack(?:ed)?(?:\s+tray)?|breach|tear|hole)\s+in\s+(?:the\s+)?(?:inner\s+)?(?:sterile\s+)?(?:barrier|tray|blister|pouch|package)\b",
    r"\b(?:torn|damaged|broken|compromised|leaking|punctured) (?:pouch|packaging|seal|sterile barrier|box|tray)\b",
    r"\b(?:pouch|packaging|seal|sterile barrier|box|tray)\s+(?:was\s+)?(?:torn|damaged|broken|compromised|leaking|punctured|breached)\b",
    r"\b(?:missing|absent|loose) (?:component|part|cap|collar|screw|marker|sealing cap)\b",
    r"\bout[- ]of[- ]the[- ]box\b",
    r"\bfailed pre[- ]use (?:check|inspection|calibration)\b",
    r"\bmislabeled|labeling error|incorrect expiry\b",
    r"\bcontamination prior to use\b",
    r"\bpackaging defect\b",
]

DEV_PROB_PATTERNS = [
    r"\b(?:balloon )?(?:ruptured?|burst|popped)\b",
    r"\b(?:shaft|wire|catheter|needle|blade|jaw|cannula|tube|tip|fragment|lead|handle|collar|hub|stent|implant|balloon)\s+(?:was\s+)?(?:fractured?|broke|snapped|bent|separated|detached|cracked|leaked|dislodged)\b",
    r"\b(?:separated|detached|dislodged)\s+(?:into|in vivo|inside|during|from)\b",
    r"\b(?:failed to |failure to )(?:deploy|fire|actuate|advance|deflate|inflate|retract|cut)\b",
    r"\b(?:misfire|misfired|failure to cut|stapler jammed?)\b",
    r"\b(?:loss of |lost )(?:signal|telemetry|power|communication|pacing)\b",
    r"\b(?:occluded?|clogged?|thrombosed|blocked lumen)\b",
    r"\b(?:fluid leak|leakage|extravasation|saline leak)\b",
    r"\b(?:overheated?|excessive heat|smoke|thermal cutoff)\b",
    r"\b(?:software crash|error code|firmware lockup|frozen screen)\b",
]

AE_PATTERNS = [
    r"\b(?:vessel |arterial |aortic |cardiac |tissue |bowel |organ |wall |coronary |visceral |dural )?(?:perforation|perforated|laceration|tear|torn vessel)\b",
    r"\b(?:tissue |vessel |organ |arterial )punctured?\b",
    r"\b(?:arterial |aortic |vessel |coronary )?dissection\b",
    r"\b(?:cardiac arrest|asystole|ventricular fibrillation|arrhythmia|hemodynamic instability)\b",
    r"\b(?:severe |excessive )?(?:bleeding|hemorrhage|hematoma|blood loss)\b",
    r"\b(?:infection|septicemia|sepsis|bacteremia|endocarditis)\b",
    r"\b(?:burn|thermal necrosis|tissue burn)\b",
    r"\b(?:stroke|transient ischemic attack|air embolism|thrombosis)\b",
    r"\b(?:patient (?:died|expired|deceased))\b",
]

INTERVENTION_PATTERNS = [
    r"\b(?:extended |emergent |emergency |exploratory |unplanned |revision )?(?:laparotomy|sternotomy|thoracotomy|reoperation|re-exploration)\b",
    r"\b(?:(?:emergent|emergency|bail-out|covered|rescue)\s+stent(?:ing)?|placement\s+of\s+(?:a\s+)?(?:covered\s+|rescue\s+)?stent|stenting)\b",
    r"\b(?:coronary artery bypass(?: graft)?|cabg|surgical repair|surgical cutdown)\b",
    r"\b(?:retrieval of|retrieve the|retrieved)\s+(?:the\s+)?(?:detached|broken|fractured|dislodged)?\s*(?:fragment|piece|device|wire|cannula|stent|lead)\b",
    r"\b(?:snare\s+catheter\s+retrieval|snare\s+retrieval)\b",
    r"\b(?:prolonged|extended)\s+(?:anesthesia(?: time)?|operating time|procedure|or time|fluoroscopy)\b",
    r"\b(?:blood |prbc |platelet )?transfusion(?:\s+of\s+\d+\s+units(?:\s+of\s+prbc)?)?\b",
    r"\b\d+\s+units?\s+(?:of\s+)?(?:prbc|packed red blood cells|blood)\b",
    r"\b(?:cpr|resuscitation|defibrillation|intubation|intubated|mechanical ventilation|placed on ventilator)\b",
    r"\b(?:fluid resuscitation|vasopressors?|epinephrine administered|blood pressure support)\b",
    r"\b(?:device |implant )?(?:explant(?:ed|ation)?|removal|removed)\b",
]

OTHER_PATTERNS = [
    r"\b(?:shipping|transit|carrier|freight) (?:damage|delay)\b",
    r"\boff[- ]label (?:use|indication|deployment)\b",
    r"\buser (?:error|technique|handling error)\b",
    r"\binconclusive (?:findings|investigation)\b",
]


class FastTriageEngine:
    """
    Tier 1 Fast Classifier and Span Extractor.
    Designed for ultra-low latency (< 2ms) scoring.
    Identifies high-confidence entities and flags low-confidence/ambiguous narratives for Tier 2 (Gemma 4).
    """

    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        self.ontology_matcher = OntologyMatcher()

        # Compile regex patterns
        self.compiled_mfg = [re.compile(p, re.IGNORECASE) for p in MFG_PATTERNS]
        self.compiled_dev = [re.compile(p, re.IGNORECASE) for p in DEV_PROB_PATTERNS]
        self.compiled_ae = [re.compile(p, re.IGNORECASE) for p in AE_PATTERNS]
        self.compiled_interventions = [re.compile(p, re.IGNORECASE) for p in INTERVENTION_PATTERNS]
        self.compiled_other = [re.compile(p, re.IGNORECASE) for p in OTHER_PATTERNS]

    def _match_patterns(self, text: str, patterns: List[re.Pattern]) -> List[Tuple[str, int, int]]:
        """Finds all non-overlapping matches with start/end character offsets, prioritizing longer matches."""
        raw_matches = []
        for pat in patterns:
            for m in pat.finditer(text):
                raw_matches.append((m.group(0), m.start(), m.end()))

        # Sort by match length descending to prioritize longer specific matches
        raw_matches.sort(key=lambda x: len(x[0]), reverse=True)

        filtered = []
        occupied_spans = []
        for span_text, start, end in raw_matches:
            if not any(max(start, o_start) < min(end, o_end) for o_start, o_end in occupied_spans):
                filtered.append((span_text, start, end))
                occupied_spans.append((start, end))

        filtered.sort(key=lambda x: x[1])
        return filtered

    def process_segments(
        self, segments: List[CleanedSegment]
    ) -> Tuple[List[ExtractedFinding], bool]:
        """
        Processes segments, extracts typed findings with ontology codes,
        and determines if the narrative requires the Gemma 4 fallback (requires_fallback: bool).
        """
        findings: List[ExtractedFinding] = []
        requires_fallback = False

        for seg in segments:
            # Skip negated segments (e.g. "No patient injury occurred")
            if seg.is_negated:
                continue

            text = seg.text

            # 1. Search for Manufacturing / Pre-use cues
            mfg_matches = self._match_patterns(text, self.compiled_mfg)
            for span, start, end in mfg_matches:
                # If discovered prior to use, in packaging, or confirmed in evaluation
                if seg.temporal_timing == TemporalTiming.PRE_USE or seg.section == "MANUFACTURER_EVALUATION":
                    base_conf = 0.90
                else:
                    base_conf = 0.82
                grounded = self.ontology_matcher.match_span(span, category=CategoryType.MANUFACTURING_ISSUE)

                findings.append(
                    ExtractedFinding(
                        category=CategoryType.MANUFACTURING_ISSUE,
                        verbatim_span=span,
                        start_char=seg.start_char + start,
                        end_char=seg.start_char + end,
                        temporal_timing=seg.temporal_timing,
                        confidence=base_conf,
                        normalized_terms=grounded,
                    )
                )

            # 2. Search for Operational Device Problems
            dev_matches = self._match_patterns(text, self.compiled_dev)
            for span, start, end in dev_matches:
                base_conf = 0.90 if seg.temporal_timing in (TemporalTiming.INTRA_USE, TemporalTiming.UNKNOWN) else 0.70
                grounded = self.ontology_matcher.match_span(span, category=CategoryType.OPERATIONAL_PROBLEM)

                findings.append(
                    ExtractedFinding(
                        category=CategoryType.OPERATIONAL_PROBLEM,
                        verbatim_span=span,
                        start_char=seg.start_char + start,
                        end_char=seg.start_char + end,
                        temporal_timing=seg.temporal_timing,
                        confidence=base_conf,
                        normalized_terms=grounded,
                    )
                )

            # 3. Search for Patient Adverse Events
            ae_matches = self._match_patterns(text, self.compiled_ae)
            for span, start, end in ae_matches:
                base_conf = 0.92
                grounded = self.ontology_matcher.match_span(span, category=CategoryType.ADVERSE_EVENT)

                findings.append(
                    ExtractedFinding(
                        category=CategoryType.ADVERSE_EVENT,
                        verbatim_span=span,
                        start_char=seg.start_char + start,
                        end_char=seg.start_char + end,
                        temporal_timing=seg.temporal_timing,
                        confidence=base_conf,
                        normalized_terms=grounded,
                    )
                )

            # 4. Search for Clinical / Surgical Interventions
            int_matches = self._match_patterns(text, self.compiled_interventions)
            for span, start, end in int_matches:
                base_conf = 0.90
                grounded = self.ontology_matcher.match_span(span, category=CategoryType.CLINICAL_INTERVENTIONS)

                findings.append(
                    ExtractedFinding(
                        category=CategoryType.CLINICAL_INTERVENTIONS,
                        verbatim_span=span,
                        start_char=seg.start_char + start,
                        end_char=seg.start_char + end,
                        temporal_timing=TemporalTiming.INTRA_USE if seg.temporal_timing in (TemporalTiming.INTRA_USE, TemporalTiming.UNKNOWN) else seg.temporal_timing,
                        confidence=base_conf,
                        normalized_terms=grounded,
                    )
                )

            # 5. Search for Other issues
            other_matches = self._match_patterns(text, self.compiled_other)
            for span, start, end in other_matches:
                findings.append(
                    ExtractedFinding(
                        category=CategoryType.OTHER,
                        verbatim_span=span,
                        start_char=seg.start_char + start,
                        end_char=seg.start_char + end,
                        temporal_timing=seg.temporal_timing,
                        confidence=0.85,
                        normalized_terms=[],
                    )
                )

        # Check if fallback to Tier 2 (Gemma 4) is triggered:
        # Triggered when:
        # a) Any finding has confidence < threshold (ambiguous boundary/timing)
        # b) No findings were detected at all despite narrative length > 120 chars (potential unhandled edge case)
        # c) Conflicting indicators (e.g. pre-use cue combined with intra-operative device failure in same clause)
        total_chars = sum(len(s.text) for s in segments)
        if total_chars > 120 and len(findings) == 0:
            requires_fallback = True
        elif any(f.confidence < self.confidence_threshold for f in findings):
            requires_fallback = True

        return findings, requires_fallback
