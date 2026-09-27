"""
Unit tests for the top-level Spotlight public API and convenience functions.
"""

import pytest
import spotlight
from spotlight import (
    MAUDERecordInput,
    MAUDEExtractionOutput,
    CategoryType,
    ExecutionPath,
    SpotlightExtractor,
)


def test_top_level_extract_string():
    narrative = "The catheter balloon burst during inflation causing arterial dissection."
    result = spotlight.extract(narrative)

    assert isinstance(result, MAUDEExtractionOutput)
    assert len(result.operational_problems) > 0
    assert len(result.adverse_events) > 0
    assert result.operational_problems[0].normalized_terms[0].code == "A0414"
    assert result.adverse_events[0].normalized_terms[0].code == "E2401"
    assert result.execution_path == ExecutionPath.FAST_PATH_MODERNBERT
    assert result.processing_time_ms > 0


def test_top_level_extract_record_object():
    record = MAUDERecordInput(
        mdr_report_key="TEST-API-001",
        brand_name="TestCatheter",
        product_code="LIT",
        event_type="Injury",
        narrative_text="During surgery, the introducer sheath hub cracked causing severe blood leak.",
    )
    result = spotlight.extract(record)

    assert result.mdr_report_key == "TEST-API-001"
    assert len(result.operational_problems) > 0
    assert result.operational_problems[0].affected_component is not None


def test_top_level_extract_batch():
    batch = [
        "Prior to use, dark particulate matter was observed in the packaging.",
        "The stapler misfired and demonstrated failure to cut the tissue.",
        "Routine procedure completed with no adverse events or device problems.",
    ]
    results = spotlight.extract_batch(batch)

    assert len(results) == 3
    assert len(results[0].manufacturing_issues) > 0
    assert len(results[1].operational_problems) >= 2
    # Negated segment: no findings
    assert len(results[2].adverse_events) == 0
    assert len(results[2].operational_problems) == 0


def test_edge_case_empty_and_whitespace_narratives():
    empty_res = spotlight.extract("")
    assert isinstance(empty_res, MAUDEExtractionOutput)
    assert len(empty_res.operational_problems) == 0

    whitespace_res = spotlight.extract("   \n\t   ")
    assert isinstance(whitespace_res, MAUDEExtractionOutput)
    assert len(whitespace_res.operational_problems) == 0


def test_edge_case_unicode_and_special_characters():
    unicode_narrative = (
        "EVENT: At 37.5°C & 12 atm, the balloon burst (ΔP > 5 atm). "
        "Patient experienced hemorrhage & hypotension; 2 units of PRBC transfused."
    )
    res = spotlight.extract(unicode_narrative)
    assert len(res.operational_problems) > 0
    assert len(res.clinical_interventions) > 0


def test_spotlight_extractor_alias():
    extractor = SpotlightExtractor(confidence_threshold=0.85, use_mock_slm=True)
    res = extractor.process_narrative("The delivery wire broke during advancement.")
    assert len(res.operational_problems) > 0
    assert res.operational_problems[0].affected_component is not None
