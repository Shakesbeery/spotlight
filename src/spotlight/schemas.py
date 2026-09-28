"""
Data contracts and schema definitions for the FDA MAUDE extraction pipeline.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CategoryType(str, Enum):
    OPERATIONAL_PROBLEM = "OPERATIONAL_PROBLEM"          # Failure mode affecting device during normal operation
    MANUFACTURING_ISSUE = "MANUFACTURING_ISSUE"          # Pre-use issue (packaging, debris, out-of-box defect)
    ADVERSE_EVENT = "ADVERSE_EVENT"                      # Clinical injury, physiological harm to patient
    CLINICAL_INTERVENTIONS = "CLINICAL_INTERVENTIONS"    # Medical / surgical rescue actions, prolonged anesthesia, etc.
    OTHER = "OTHER"                                      # Off-label use, transit damage, ambiguous findings


class TemporalTiming(str, Enum):
    PRE_USE = "PRE_USE"            # Before contact with patient (inspection, packaging, prep)
    INTRA_USE = "INTRA_USE"        # During operation/procedure
    POST_USE = "POST_USE"          # Post-procedure discovery, explant, delayed adverse event
    UNKNOWN = "UNKNOWN"


class ExecutionPath(str, Enum):
    FAST_PATH_MODERNBERT = "FAST_PATH_MODERNBERT"
    SLM_FALLBACK_GEMMA4 = "SLM_FALLBACK_GEMMA4"
    HYBRID_VERIFIED = "HYBRID_VERIFIED"


class GroundedOntologyTerm(BaseModel):
    """Normalized medical device or clinical concept code (IMDRF / MedDRA)."""
    code: str = Field(..., description="Standard code, e.g., A0401, C1901, E0506")
    preferred_term: str = Field(..., description="Official terminology preferred term")
    ontology: str = Field(..., description="Ontology name, e.g., IMDRF_Annex_A, IMDRF_Annex_C, IMDRF_Annex_E, MedDRA")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic match confidence score")


class ExtractedFinding(BaseModel):
    """An individual extracted finding belonging to one of the core categories."""
    id: Optional[str] = Field(None, description="Unique finding ID (e.g. OP-1, MFG-1, INT-1)")
    category: CategoryType = Field(..., description="Target category classification")
    verbatim_span: str = Field(..., description="Exact textual span from the narrative")
    start_char: Optional[int] = Field(None, ge=0, description="Start character offset in cleaned narrative")
    end_char: Optional[int] = Field(None, ge=0, description="End character offset in cleaned narrative")
    temporal_timing: TemporalTiming = Field(default=TemporalTiming.UNKNOWN, description="Timing relative to device use")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    affected_component: Optional[str] = Field(None, description="Specific device component linked to failure (e.g. sheath hub, tulip collar, balloon)")
    failure_mode: Optional[str] = Field(None, description="Granular physical failure mechanism")
    aliases: List[str] = Field(default_factory=list, description="Coreferent mentions, acronyms, or textual aliases (e.g. CABG)")
    triggered_by: Optional[str] = Field(None, description="ID of cause finding triggering this consequence")
    normalized_terms: List[GroundedOntologyTerm] = Field(
        default_factory=list,
        description="Ranked normalized ontology concepts (IMDRF/MedDRA)"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata such as affected component, mechanism, or clinical severity"
    )

    model_config = ConfigDict(frozen=True)


class MAUDERecordInput(BaseModel):
    """Raw or structured input record representing an FDA MAUDE report."""
    mdr_report_key: str = Field(default="ADHOC-RECORD", description="FDA unique report identifier (MDR Key)")
    brand_name: Optional[str] = Field(None, description="Trade/Brand name of device")
    product_code: Optional[str] = Field(None, description="FDA 3-letter product code (e.g., DQX, KRC)")
    event_type: Optional[str] = Field(None, description="FDA event type: Malfunction, Injury, Death, Other")
    narrative_text: str = Field(..., description="Free-text narrative (FOI text / evaluation text)")


class MAUDEExtractionOutput(BaseModel):
    """Complete extraction result for a MAUDE record."""
    mdr_report_key: str = Field(..., description="Report identifier")
    brand_name: Optional[str] = Field(None, description="Trade/Brand name of device")
    product_code: Optional[str] = Field(None, description="FDA 3-letter product code (e.g., DQX, KRC)")
    event_type: Optional[str] = Field(None, description="FDA event type: Malfunction, Injury, Death, Other")
    operational_problems: List[ExtractedFinding] = Field(
        default_factory=list,
        description="Modes of failure affecting the device during normal operation"
    )
    manufacturing_issues: List[ExtractedFinding] = Field(
        default_factory=list,
        description="Problems affecting devices before use (e.g., debris, packaging damage)"
    )
    adverse_events: List[ExtractedFinding] = Field(
        default_factory=list,
        description="Adverse events / patient harm during normal operation"
    )
    clinical_interventions: List[ExtractedFinding] = Field(
        default_factory=list,
        description="Medical, surgical, or procedural interventions required to rescue/treat patient (IMDRF Annex F)"
    )
    other: List[ExtractedFinding] = Field(
        default_factory=list,
        description="Problems or AEs not falling into the main classes"
    )
    execution_path: ExecutionPath = Field(..., description="Model tier that processed this record")
    processing_time_ms: float = Field(..., ge=0.0, description="Latency in milliseconds")
    cleaned_narrative: str = Field(..., description="Narrative after boilerplate removal")
