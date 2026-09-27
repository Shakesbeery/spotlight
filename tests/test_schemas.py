"""
Unit tests for data contracts and Pydantic schemas.
"""

import pytest
from spotlight.schemas import (
    CategoryType,
    TemporalTiming,
    ExecutionPath,
    ExtractedFinding,
    GroundedOntologyTerm,
    MAUDERecordInput,
    MAUDEExtractionOutput,
)


def test_schema_valid_instantiation():
    term = GroundedOntologyTerm(
        code="A0401",
        preferred_term="Balloon Burst / Rupture",
        ontology="IMDRF_Annex_A",
        similarity_score=0.95
    )
    assert term.code == "A0401"
    assert term.similarity_score == 0.95

    finding = ExtractedFinding(
        category=CategoryType.OPERATIONAL_PROBLEM,
        verbatim_span="balloon ruptured at 14 atm",
        start_char=10,
        end_char=36,
        temporal_timing=TemporalTiming.INTRA_USE,
        confidence=0.92,
        normalized_terms=[term],
        details={"pressure_atm": 14}
    )
    assert finding.category == CategoryType.OPERATIONAL_PROBLEM
    assert len(finding.normalized_terms) == 1

    output = MAUDEExtractionOutput(
        mdr_report_key="MDR-123456",
        operational_problems=[finding],
        manufacturing_issues=[],
        adverse_events=[],
        other=[],
        execution_path=ExecutionPath.FAST_PATH_MODERNBERT,
        processing_time_ms=1.45,
        cleaned_narrative="Device balloon ruptured at 14 atm during inflation."
    )
    assert output.mdr_report_key == "MDR-123456"
    assert len(output.operational_problems) == 1
    assert output.execution_path == ExecutionPath.FAST_PATH_MODERNBERT


def test_schema_confidence_bounds():
    with pytest.raises(ValueError):
        ExtractedFinding(
            category=CategoryType.ADVERSE_EVENT,
            verbatim_span="vascular dissection",
            confidence=1.5  # Invalid: > 1.0
        )

    with pytest.raises(ValueError):
        GroundedOntologyTerm(
            code="E0506",
            preferred_term="Hemorrhage",
            ontology="IMDRF_Annex_E",
            similarity_score=-0.1  # Invalid: < 0.0
        )
