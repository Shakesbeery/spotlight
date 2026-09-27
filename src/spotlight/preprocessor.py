"""
Preprocessor module for MAUDE FDA records.
Handles boilerplate removal, section parsing, temporal tagging, and negation cue detection.
"""

import re
from typing import List, Dict, Tuple, Optional
from spotlight.schemas import TemporalTiming


# Standard recurring FDA and manufacturer legal boilerplate patterns
BOILERPLATE_PATTERNS = [
    r"(?i)this report is being submitted pursuant to (?:21\s*cfr|fda regulations)[^\.\n]*[\.\n]?",
    r"(?i)the submission of this report does not constitute an admission (?:that|of)[^\.\n]*[\.\n]?",
    r"(?i)information included in this report does not represent a conclusion by fda[^\.\n]*[\.\n]?",
    r"(?i)no conclusion can be drawn at this time regarding the root cause[^\.\n]*[\.\n]?",
    r"(?i)the device has not been returned to the manufacturer for evaluation[^\.\n]*[\.\n]?",
    r"(?i)if additional information becomes available, a supplemental report will be submitted[^\.\n]*[\.\n]?",
    r"(?i)investigation is ongoing and results will be provided in a supplemental report[^\.\n]*[\.\n]?",
    r"\(b\)\([46]\)",  # FDA redaction tokens (b)(4) and (b)(6)
    r"(?i)\*{3,}\s*(?:redacted|confidential)\s*\*{3,}",
]

# Temporal indicators
PRE_USE_CUES = [
    r"\bprior to (?:use|patient contact|surgery|procedure|implantation)\b",
    r"\bupon (?:opening|inspection|unpacking|unboxing)\b",
    r"\bout of the box\b",
    r"\bin the (?:package|packaging|sterile pouch|box|tray)\b",
    r"\bbefore (?:use|patient contact|insertion|surgery)\b",
    r"\bduring prep(?:aration)?\b",
    r"\bsterile barrier\b",
    r"\bfailed pre-check\b",
]

INTRA_USE_CUES = [
    r"\bduring (?:the )?(?:procedure|surgery|operation|inflation|advancement|deployment|ablation)\b",
    r"\bintra-?operatively\b",
    r"\bwhile (?:inflating|advancing|deploying|operating|using|cutting|cauterizing)\b",
    r"\bin vivo\b",
    r"\bupon (?:insertion|activation|deployment|contact with tissue)\b",
    r"\bpatient was on (?:the table|bypass)\b",
]

POST_USE_CUES = [
    r"\bpost-?operatively\b",
    r"\bfollowing the procedure\b",
    r"\bat (?:the )?(?:follow-up|1-month|delayed)\b",
    r"\bafter (?:procedure completion|removal|explant)\b",
    r"\bupon explant(?:ation)?\b",
    r"\bhours after\b",
    r"\bdays later\b",
]

# Negation triggers
NEGATION_PATTERNS = [
    r"\bno (?:patient )?(?:injury|harm|adverse event|complication|defect|failure|issue|particulate|damage)\b",
    r"\bno patient(?: was)? (?:involved|contact)\b",
    r"\bdenie[sd] any\b",
    r"\bwithout (?:any )?(?:patient )?(?:harm|injury|consequence|incident)\b",
    r"\bwas not (?:used|implanted|damaged|defective|involved)\b",
    r"\bnot observed\b",
    r"\bfree of (?:debris|particulate|contamination)\b",
    r"\buninjured\b",
]


class CleanedSegment:
    """Represents a segmented clause or sentence with extracted context metadata."""
    def __init__(
        self,
        text: str,
        section: str,
        start_char: int,
        end_char: int,
        temporal_timing: TemporalTiming = TemporalTiming.UNKNOWN,
        is_negated: bool = False,
    ):
        self.text = text
        self.section = section
        self.start_char = start_char
        self.end_char = end_char
        self.temporal_timing = temporal_timing
        self.is_negated = is_negated

    def __repr__(self) -> str:
        return (
            f"CleanedSegment(section='{self.section}', timing={self.temporal_timing.value}, "
            f"negated={self.is_negated}, text='{self.text[:40]}...')"
        )


class MAUDEPreprocessor:
    """
    Cleans raw FDA narrative texts, removes boilerplate disclaimers,
    splits into structured sections, and segments text with temporal and negation markers.
    """

    def __init__(self):
        self.compiled_boilerplate = [re.compile(p, re.DOTALL) for p in BOILERPLATE_PATTERNS]
        self.compiled_pre_use = [re.compile(p, re.IGNORECASE) for p in PRE_USE_CUES]
        self.compiled_intra_use = [re.compile(p, re.IGNORECASE) for p in INTRA_USE_CUES]
        self.compiled_post_use = [re.compile(p, re.IGNORECASE) for p in POST_USE_CUES]
        self.compiled_negations = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]

    def strip_boilerplate(self, text: str) -> str:
        """Removes legal and regulatory boilerplate text and cleans up whitespace."""
        cleaned = text
        for pattern in self.compiled_boilerplate:
            cleaned = pattern.sub(" ", cleaned)
        # Collapse multiple spaces and newlines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parses common section headers present in MAUDE reports (e.g. EVENT DESCRIPTION vs EVALUATION).
        Returns a dictionary mapping section names to text.
        """
        section_markers = [
            (r"(?i)\b(?:event description|description of event|report narrative):\s*", "EVENT_DESCRIPTION"),
            (r"(?i)\b(?:manufacturer evaluation|investigation summary|manufacturer narrative|device evaluation):\s*", "MANUFACTURER_EVALUATION"),
            (r"(?i)\b(?:additional manufacturer narrative|remedial action|corrective action):\s*", "ADDITIONAL_NARRATIVE"),
        ]

        # Find marker positions
        found = []
        for pat, sec_name in section_markers:
            for match in re.finditer(pat, text):
                found.append((match.start(), match.end(), sec_name))

        if not found:
            return {"EVENT_DESCRIPTION": text}

        # Sort by start offset
        found.sort(key=lambda x: x[0])
        sections = {}

        # Content before first header
        if found[0][0] > 0:
            prefix = text[: found[0][0]].strip()
            if prefix:
                sections["EVENT_DESCRIPTION"] = prefix

        for i, (start, end, sec_name) in enumerate(found):
            next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
            sec_text = text[end:next_start].strip()
            if sec_name in sections:
                sections[sec_name] += "\n" + sec_text
            else:
                sections[sec_name] = sec_text

        return sections

    def infer_temporal_timing(self, text: str) -> TemporalTiming:
        """Detects whether clause discusses pre-use, intra-use, or post-use."""
        for pattern in self.compiled_pre_use:
            if pattern.search(text):
                return TemporalTiming.PRE_USE
        for pattern in self.compiled_intra_use:
            if pattern.search(text):
                return TemporalTiming.INTRA_USE
        for pattern in self.compiled_post_use:
            if pattern.search(text):
                return TemporalTiming.POST_USE
        return TemporalTiming.UNKNOWN

    def check_negation(self, text: str) -> bool:
        """Detects if sentence explicitly states absence of injury/defect."""
        for pattern in self.compiled_negations:
            if pattern.search(text):
                return True
        return False

    def segment_text(self, text: str) -> List[CleanedSegment]:
        """
        Cleans, parses sections, and breaks narrative down into typed clauses/segments.
        """
        cleaned = self.strip_boilerplate(text)
        sections = self.parse_sections(cleaned)
        segments: List[CleanedSegment] = []

        global_char_offset = 0
        for sec_name, sec_content in sections.items():
            # Regex sentence boundary splitting respecting abbreviations (e.g. Dr., St., etc.)
            raw_sentences = re.split(r"(?<=[.!?])\s+", sec_content)
            current_offset = 0

            for sent in raw_sentences:
                sent = sent.strip()
                if not sent:
                    continue

                sent_start = sec_content.find(sent, current_offset)
                sent_end = sent_start + len(sent) if sent_start != -1 else current_offset + len(sent)
                current_offset = sent_end

                timing = self.infer_temporal_timing(sent)
                is_negated = self.check_negation(sent)

                segments.append(
                    CleanedSegment(
                        text=sent,
                        section=sec_name,
                        start_char=sent_start,
                        end_char=sent_end,
                        temporal_timing=timing,
                        is_negated=is_negated,
                    )
                )

        return segments
