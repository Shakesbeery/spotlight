"""
Unit tests for fast triage and candidate span extraction.
"""

from spotlight.preprocessor import MAUDEPreprocessor
from spotlight.fast_triage import FastTriageEngine
from spotlight.schemas import CategoryType


def test_fast_triage_operational_and_ae():
    preprocessor = MAUDEPreprocessor()
    triage = FastTriageEngine(confidence_threshold=0.80)

    text = "During surgery, the balloon burst under pressure. The patient sustained severe bleeding."
    segments = preprocessor.segment_text(text)
    findings, requires_fallback = triage.process_segments(segments)

    categories = [f.category for f in findings]
    assert CategoryType.OPERATIONAL_PROBLEM in categories
    assert CategoryType.ADVERSE_EVENT in categories
    assert not requires_fallback


def test_fast_triage_manufacturing_pre_use():
    preprocessor = MAUDEPreprocessor()
    triage = FastTriageEngine(confidence_threshold=0.80)

    text = "Prior to use, the technician discovered foreign debris inside the sterile pouch."
    segments = preprocessor.segment_text(text)
    findings, requires_fallback = triage.process_segments(segments)

    assert len(findings) >= 1
    mfg_finding = [f for f in findings if f.category == CategoryType.MANUFACTURING_ISSUE][0]
    assert mfg_finding.confidence >= 0.80
    assert not requires_fallback


def test_fast_triage_negated_ae_ignored():
    preprocessor = MAUDEPreprocessor()
    triage = FastTriageEngine(confidence_threshold=0.80)

    text = "The wire detached during deployment. No patient injury occurred."
    segments = preprocessor.segment_text(text)
    findings, _ = triage.process_segments(segments)

    # Detached is an operational problem
    assert any(f.category == CategoryType.OPERATIONAL_PROBLEM for f in findings)
    # Negated AE should NOT create an adverse event finding
    assert not any(f.category == CategoryType.ADVERSE_EVENT for f in findings)


def test_fast_triage_fallback_triggered_on_unrecognized_long_text():
    preprocessor = MAUDEPreprocessor()
    triage = FastTriageEngine(confidence_threshold=0.80)

    # Long clinical narrative with complex terminology not caught by standard fast patterns
    text = (
        "The clinician was undertaking a complex structural intervention. "
        "Unexpected hemodynamics were observed following an atypical anatomy approach, "
        "and extensive multidisciplinary discussions were recorded."
    )
    segments = preprocessor.segment_text(text)
    findings, requires_fallback = triage.process_segments(segments)

    # Since text > 120 chars and no findings were detected, fallback should trigger
    assert len(findings) == 0
    assert requires_fallback is True


def test_fast_triage_mixed_sentence_negation_preserves_positive_finding():
    """Verifies that within a single sentence containing both an affirmative malfunction
    and a negated injury, the affirmative finding is extracted while the negated injury is suppressed."""
    preprocessor = MAUDEPreprocessor()
    triage = FastTriageEngine(confidence_threshold=0.80)

    # In one sentence: positive malfunction ("fractured") + contrastive negated AE ("no patient injury")
    text = "The balloon catheter fractured during deployment, but no patient injury or bleeding occurred."
    segments = preprocessor.segment_text(text)
    findings, _ = triage.process_segments(segments)

    # Operational problem (fractured) should be captured
    assert any(f.category == CategoryType.OPERATIONAL_PROBLEM for f in findings)
    # The negated adverse events (patient injury, bleeding) should NOT be captured
    assert not any(f.category == CategoryType.ADVERSE_EVENT for f in findings)

