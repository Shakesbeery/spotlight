"""
Spotlight: Production-grade NLP Information Extraction System for FDA MAUDE narratives.
Specialized for operational device problems, manufacturing/pre-use defects, clinical adverse events,
and health impact interventions with granular IMDRF (Annexes A, C, E, F) and MedDRA ontology grounding.
"""

from typing import Union, List, Optional
from spotlight.schemas import (
    CategoryType,
    TemporalTiming,
    ExecutionPath,
    GroundedOntologyTerm,
    ExtractedFinding,
    MAUDERecordInput,
    MAUDEExtractionOutput,
)
from spotlight.pipeline import MAUDEExtractionPipeline, SpotlightExtractor
from spotlight.preprocessor import MAUDEPreprocessor
from spotlight.ontology import OntologyMatcher
from spotlight.fast_triage import FastTriageEngine
from spotlight.component_linker import ComponentLinker, EntityResolver
from spotlight.data_fetcher import MAUDEDataFetcher

__version__ = "0.2.0"

_DEFAULT_EXTRACTOR: Optional[SpotlightExtractor] = None


def get_default_extractor() -> SpotlightExtractor:
    """Returns a singleton instance of the default high-throughput Spotlight extractor."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SpotlightExtractor(confidence_threshold=0.80, use_mock_slm=True)
    return _DEFAULT_EXTRACTOR


def extract(
    narrative: Union[str, MAUDERecordInput],
    mdr_report_key: str = "ADHOC-RECORD",
    brand_name: Optional[str] = None,
    product_code: Optional[str] = None,
    event_type: str = "Unknown",
) -> MAUDEExtractionOutput:
    """
    Extracts structured clinical findings and grounded IMDRF/MedDRA terms from a narrative string or record.

    Example:
        >>> import spotlight
        >>> result = spotlight.extract("Balloon burst under pressure causing coronary dissection.")
        >>> print(result.operational_problems[0].failure_mode)
    """
    extractor = get_default_extractor()
    if isinstance(narrative, str):
        return extractor.process_narrative(
            narrative_text=narrative,
            mdr_report_key=mdr_report_key,
            brand_name=brand_name,
            product_code=product_code,
            event_type=event_type,
        )
    return extractor.process_record(narrative)


def extract_batch(
    records: List[Union[str, MAUDERecordInput]]
) -> List[MAUDEExtractionOutput]:
    """
    Processes a batch of narratives or records with high throughput.

    Example:
        >>> import spotlight
        >>> results = spotlight.extract_batch(["Report 1 text...", "Report 2 text..."])
    """
    extractor = get_default_extractor()
    return extractor.process_batch(records)


__all__ = [
    "__version__",
    "extract",
    "extract_batch",
    "get_default_extractor",
    "SpotlightExtractor",
    "MAUDEExtractionPipeline",
    "MAUDERecordInput",
    "MAUDEExtractionOutput",
    "ExtractedFinding",
    "CategoryType",
    "TemporalTiming",
    "ExecutionPath",
    "GroundedOntologyTerm",
    "MAUDEPreprocessor",
    "OntologyMatcher",
    "FastTriageEngine",
    "ComponentLinker",
    "EntityResolver",
    "MAUDEDataFetcher",
]
