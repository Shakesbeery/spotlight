"""
Unit tests for MAUDE preprocessor and text segmentation.
"""

from spotlight.preprocessor import MAUDEPreprocessor
from spotlight.schemas import TemporalTiming


def test_strip_boilerplate():
    preprocessor = MAUDEPreprocessor()
    raw_text = (
        "During surgery the catheter fractured. (b)(4) "
        "THIS REPORT IS BEING SUBMITTED PURSUANT TO 21 CFR PART 803. "
        "The submission of this report does not constitute an admission that the device caused the event. "
        "Patient required revision surgery."
    )
    cleaned = preprocessor.strip_boilerplate(raw_text)
    assert "(b)(4)" not in cleaned
    assert "21 CFR" not in cleaned
    assert "admission" not in cleaned
    assert "During surgery the catheter fractured." in cleaned
    assert "Patient required revision surgery." in cleaned


def test_section_parsing():
    preprocessor = MAUDEPreprocessor()
    raw_text = (
        "EVENT DESCRIPTION: The delivery system detached inside the patient during catheter advancement. "
        "MANUFACTURER EVALUATION: The returned device was inspected under microscopy and showed tensile failure."
    )
    sections = preprocessor.parse_sections(raw_text)
    assert "EVENT_DESCRIPTION" in sections
    assert "MANUFACTURER_EVALUATION" in sections
    assert "detached inside the patient" in sections["EVENT_DESCRIPTION"]
    assert "microscopy" in sections["MANUFACTURER_EVALUATION"]


def test_temporal_tagging():
    preprocessor = MAUDEPreprocessor()

    pre_use = "Prior to use, the technician noted foreign particulate inside the sterile blister pack."
    assert preprocessor.infer_temporal_timing(pre_use) == TemporalTiming.PRE_USE

    intra_use = "Intraoperatively, while inflating the balloon, a sudden drop in pressure was observed."
    assert preprocessor.infer_temporal_timing(intra_use) == TemporalTiming.INTRA_USE

    post_use = "At the 1-month follow-up, the patient presented with local infection."
    assert preprocessor.infer_temporal_timing(post_use) == TemporalTiming.POST_USE

    unknown = "The catheter model 450 was involved in an incident."
    assert preprocessor.infer_temporal_timing(unknown) == TemporalTiming.UNKNOWN


def test_negation_detection():
    preprocessor = MAUDEPreprocessor()

    negated_1 = "No patient injury or harm was reported as a result of this occurrence."
    assert preprocessor.check_negation(negated_1) is True

    negated_2 = "Package was inspected and confirmed free of debris."
    assert preprocessor.check_negation(negated_2) is True

    positive_ae = "Patient sustained severe arterial dissection requiring emergency intervention."
    assert preprocessor.check_negation(positive_ae) is False


def test_segment_text():
    preprocessor = MAUDEPreprocessor()
    narrative = (
        "EVENT DESCRIPTION: Prior to use, particulate was noted in the package. "
        "The device was replaced. No patient injury occurred."
    )
    segments = preprocessor.segment_text(narrative)
    assert len(segments) == 3

    assert segments[0].temporal_timing == TemporalTiming.PRE_USE
    assert "particulate was noted" in segments[0].text

    assert segments[2].is_negated is True
    assert "No patient injury" in segments[2].text
