"""
Spotlight Live FDA MAUDE Ingestion Demo.
Demonstrates querying the official openFDA API and processing real-world MDR narratives.
"""

import spotlight
from spotlight import MAUDEDataFetcher


def main():
    fetcher = MAUDEDataFetcher(data_dir="data")

    print("Querying openFDA Device Event API for live real-world records...")
    records = fetcher.fetch_openfda_records(
        limit=5,
        event_type="Injury",
        query='(mdr_text.text:"perforation" OR mdr_text.text:"dissection" OR mdr_text.text:"rupture")'
    )

    print(f"Successfully retrieved {len(records)} live records from FDA.")
    print("Processing through Spotlight NLP pipeline...\n")

    extractor = spotlight.get_default_extractor()
    for i, r in enumerate(records, 1):
        out = extractor.process_record(r)
        print("=" * 70)
        print(f"[{i}/{len(records)}] MDR Key: {r.mdr_report_key} | Device: {r.brand_name or 'N/A'}")
        print(f"Latency: {out.processing_time_ms:.2f} ms | Route: {out.execution_path.value}")

        total_findings = len(out.operational_problems) + len(out.manufacturing_issues) + len(out.adverse_events) + len(out.clinical_interventions)
        print(f"Total Findings Detected: {total_findings}")

        if out.operational_problems:
            print("  • Operational Problems:")
            for op in out.operational_problems:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in op.normalized_terms])
                print(f"    - \"{op.verbatim_span}\" -> {codes}")

        if out.adverse_events:
            print("  • Patient Adverse Events:")
            for ae in out.adverse_events:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in ae.normalized_terms])
                print(f"    - \"{ae.verbatim_span}\" -> {codes}")

        if out.clinical_interventions:
            print("  • Clinical Interventions:")
            for intv in out.clinical_interventions:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in intv.normalized_terms])
                print(f"    - \"{intv.verbatim_span}\" -> {codes}")


if __name__ == "__main__":
    main()
