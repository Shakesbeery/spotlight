"""
Spotlight Demonstration Script.
Processes sample FDA MAUDE reports exhibiting operational problems,
manufacturing issues, patient adverse events, and boilerplate text.
"""

import json
from spotlight.pipeline import MAUDEExtractionPipeline
from spotlight.schemas import MAUDERecordInput


def run_demo():
    print("=" * 80)
    print("SPOTLIGHT: FDA MAUDE NLP EXTRACTION PIPELINE DEMONSTRATION")
    print("=" * 80)

    sample_reports = [
        MAUDERecordInput(
            mdr_report_key="MDR-2026-00129",
            brand_name="AeroGlide PTCA Balloon Catheter",
            product_code="LIT",
            event_type="Injury",
            narrative_text=(
                "THIS REPORT IS BEING SUBMITTED PURSUANT TO 21 CFR PART 803. (b)(4) "
                "EVENT DESCRIPTION: Prior to use, the technician noted foreign particulate "
                "inside the sterile pouch. A replacement catheter was opened. During the "
                "angioplasty procedure, while inflating at 14 atm, the balloon ruptured suddenly. "
                "The patient experienced severe arterial dissection requiring emergency stenting. "
                "No other patient injury was reported. "
                "MANUFACTURER EVALUATION: The device was returned. Microscopy confirmed rupture along the longitudinal seam."
            ),
        ),
        MAUDERecordInput(
            mdr_report_key="MDR-2026-00130",
            brand_name="EndoGrip Laparoscopic Grasper",
            product_code="GCJ",
            event_type="Malfunction",
            narrative_text=(
                "THE SUBMISSION OF THIS REPORT DOES NOT CONSTITUTE AN ADMISSION OF DEFECT. "
                "EVENT DESCRIPTION: Out-of-the-box inspection revealed a damaged sterile barrier "
                "and compromised seal on the blister package. Device was not used on patient. "
                "No patient harm occurred."
            ),
        ),
    ]

    pipeline = MAUDEExtractionPipeline(use_mock_slm=True)

    for record in sample_reports:
        print(f"\n[Processing MDR Key: {record.mdr_report_key}] ({record.brand_name})")
        output = pipeline.process_record(record)

        print(f"Execution Path     : {output.execution_path.value}")
        print(f"Latency            : {output.processing_time_ms:.2f} ms")
        print(f"Cleaned Narrative  : {output.cleaned_narrative[:100]}...")

        print("\n--- Extracted Categories ---")
        print(f"1. Operational Problems ({len(output.operational_problems)}):")
        for f in output.operational_problems:
            codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
            print(f"   • Verbatim: \"{f.verbatim_span}\" | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

        print(f"2. Manufacturing Issues ({len(output.manufacturing_issues)}):")
        for f in output.manufacturing_issues:
            codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
            print(f"   • Verbatim: \"{f.verbatim_span}\" | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

        print(f"3. Adverse Events ({len(output.adverse_events)}):")
        for f in output.adverse_events:
            codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
            print(f"   • Verbatim: \"{f.verbatim_span}\" | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

        print(f"4. Clinical Interventions ({len(output.clinical_interventions)}):")
        for f in output.clinical_interventions:
            codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
            trig = f" -> Triggered by: {f.triggered_by}" if f.triggered_by else ""
            print(f"   • Verbatim: \"{f.verbatim_span}\"{trig} | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

        print(f"5. Other ({len(output.other)}):")
        for f in output.other:
            print(f"   • Verbatim: \"{f.verbatim_span}\"")
        print("-" * 80)


if __name__ == "__main__":
    run_demo()
