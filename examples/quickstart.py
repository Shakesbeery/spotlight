"""
Spotlight Quickstart Example.
Demonstrates extracting operational problems, manufacturing defects, adverse events,
and clinical interventions from a single medical device narrative.
"""

import spotlight


def main():
    # Example 1: Extract directly from a raw narrative string
    narrative = (
        "During laparoscopic colectomy, the surgical stapler misfired across the mesenteric vessels "
        "and demonstrated complete failure to cut. The jammed jaw caused severe tissue laceration and "
        "acute hemorrhage. The surgical team performed an emergency exploratory laparotomy to achieve hemostasis. "
        "The patient received a blood transfusion of 2 units of PRBC and required prolonged anesthesia time."
    )

    print("Analyzing narrative with Spotlight...")
    result = spotlight.extract(narrative, mdr_report_key="DEMO-001", brand_name="Articulating Stapler")

    print("\n" + "=" * 70)
    print(f"REPORT KEY  : {result.mdr_report_key}")
    print(f"ROUTE       : {result.execution_path.value}")
    print(f"LATENCY     : {result.processing_time_ms:.2f} ms")
    print("=" * 70)

    print(f"\n1. Operational Problems ({len(result.operational_problems)}):")
    for f in result.operational_problems:
        codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
        comp = f" [Component: {f.affected_component}]" if f.affected_component else ""
        print(f"   • {f.id}: \"{f.verbatim_span}\"{comp} -> IMDRF: {codes}")

    print(f"\n2. Patient Adverse Events ({len(result.adverse_events)}):")
    for f in result.adverse_events:
        codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
        print(f"   • {f.id}: \"{f.verbatim_span}\" -> IMDRF/MedDRA: {codes}")

    print(f"\n3. Clinical Interventions ({len(result.clinical_interventions)}):")
    for f in result.clinical_interventions:
        codes = ", ".join([f"{t.code} ({t.preferred_term})" for t in f.normalized_terms])
        trig = f" (Triggered by {f.triggered_by})" if f.triggered_by else ""
        print(f"   • {f.id}: \"{f.verbatim_span}\"{trig} -> IMDRF: {codes}")


if __name__ == "__main__":
    main()
