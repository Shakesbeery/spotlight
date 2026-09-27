"""
Integration and end-to-end pipeline tests.
"""

from spotlight.pipeline import MAUDEExtractionPipeline
from spotlight.schemas import MAUDERecordInput, ExecutionPath, CategoryType


def test_pipeline_fast_path_operational_and_ae():
    pipeline = MAUDEExtractionPipeline()

    record = MAUDERecordInput(
        mdr_report_key="MDR-99001",
        brand_name="UltraFlex Stent Catheter",
        product_code="NIQ",
        event_type="Injury",
        narrative_text=(
            "THIS REPORT IS BEING SUBMITTED PURSUANT TO 21 CFR PART 803. (b)(4) "
            "EVENT DESCRIPTION: During surgery, the catheter shaft fractured upon advancement. "
            "The patient experienced severe arterial dissection requiring emergency stenting. "
            "No conclusion can be drawn at this time regarding the root cause."
        )
    )

    output = pipeline.process_record(record)

    assert output.mdr_report_key == "MDR-99001"
    assert output.execution_path == ExecutionPath.FAST_PATH_MODERNBERT
    assert "21 CFR" not in output.cleaned_narrative
    assert "(b)(4)" not in output.cleaned_narrative
    assert output.processing_time_ms > 0

    # Verify operational problem
    assert len(output.operational_problems) >= 1
    op_finding = output.operational_problems[0]
    assert op_finding.category == CategoryType.OPERATIONAL_PROBLEM
    assert any(term.ontology == "IMDRF_Annex_A" for term in op_finding.normalized_terms)

    # Verify adverse event
    assert len(output.adverse_events) >= 1
    ae_finding = output.adverse_events[0]
    assert ae_finding.category == CategoryType.ADVERSE_EVENT
    assert any(term.ontology == "IMDRF_Annex_E" for term in ae_finding.normalized_terms)

    # Verify clinical intervention
    assert len(output.clinical_interventions) >= 1
    int_finding = output.clinical_interventions[0]
    assert int_finding.category == CategoryType.CLINICAL_INTERVENTIONS
    assert any(term.ontology == "IMDRF_Annex_F" for term in int_finding.normalized_terms)
    assert int_finding.triggered_by is not None


def test_pipeline_fast_path_manufacturing_issue():
    pipeline = MAUDEExtractionPipeline()

    record = MAUDERecordInput(
        mdr_report_key="MDR-99002",
        brand_name="AeroSeal Introducer",
        product_code="DYB",
        event_type="Malfunction",
        narrative_text=(
            "Prior to use, the circulating nurse noted foreign particulate inside the sterile pouch. "
            "The device was discarded and replaced. No patient injury occurred."
        )
    )

    output = pipeline.process_record(record)

    assert output.mdr_report_key == "MDR-99002"
    assert len(output.manufacturing_issues) >= 1
    mfg = output.manufacturing_issues[0]
    assert mfg.category == CategoryType.MANUFACTURING_ISSUE
    assert any(term.code == "C1901" for term in mfg.normalized_terms)

    # Negated AE should not appear in adverse_events
    assert len(output.adverse_events) == 0


def test_pipeline_slm_fallback():
    pipeline = MAUDEExtractionPipeline(use_mock_slm=True)

    # Complex narrative without standard fast keywords
    record = MAUDERecordInput(
        mdr_report_key="MDR-99003",
        brand_name="OmniWave Pulse Generator",
        product_code="LWS",
        event_type="Malfunction",
        narrative_text=(
            "The clinician observed anomalous telemetry telemetry feedback with unexpected sensor "
            "drift during prolonged pacing across atypical patient anatomical features. Extensive "
            "multidisciplinary discussions occurred and the device locked up during procedure."
        )
    )

    output = pipeline.process_record(record)

    assert output.mdr_report_key == "MDR-99003"
    assert output.execution_path == ExecutionPath.SLM_FALLBACK_GEMMA4
    assert len(output.operational_problems) >= 1


def test_pipeline_batch_processing():
    pipeline = MAUDEExtractionPipeline()

    records = [
        MAUDERecordInput(
            mdr_report_key=f"BATCH-{i}",
            narrative_text="During surgery, the balloon burst under pressure."
        )
        for i in range(5)
    ]

    results = pipeline.process_batch(records)
    assert len(results) == 5
    for r in results:
        assert len(r.operational_problems) >= 1
        assert r.execution_path == ExecutionPath.FAST_PATH_MODERNBERT
