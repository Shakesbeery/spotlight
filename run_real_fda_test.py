"""
Real FDA MAUDE Extraction Test Script.
Fetches real-world FDA MAUDE reports from the official FDA database,
runs them through the extraction pipeline, and displays structured results.
"""

import sys
import time
from spotlight.data_fetcher import MAUDEDataFetcher
from spotlight.pipeline import MAUDEExtractionPipeline
from spotlight.schemas import CategoryType


def main():
    print("=" * 80)
    print("SPOTLIGHT: REAL FDA MAUDE INGESTION & EXTRACTION TEST")
    print("=" * 80)

    fetcher = MAUDEDataFetcher(data_dir="data")
    pipeline = MAUDEExtractionPipeline(confidence_threshold=0.80, use_mock_slm=True)

    print("\n[Step 1] Fetching live diverse FDA MAUDE records across categories...")
    t0 = time.time()
    records = fetcher.fetch_diverse_real_dataset(samples_per_category=4)
    fetch_time = time.time() - t0
    print(f"-> Successfully retrieved {len(records)} real FDA records in {fetch_time:.2f}s.")

    # Save to local file
    saved_path = fetcher.save_records_to_json(records, "real_fda_records_sample.json")
    print(f"-> Saved raw FDA records to: {saved_path}")

    print("\n[Step 2] Executing Spotlight Pipeline on Real FDA Records...")
    print("-" * 80)

    total_op = 0
    total_mfg = 0
    total_ae = 0
    total_other = 0
    latencies = []

    for idx, record in enumerate(records, 1):
        print(f"\n[{idx}/{len(records)}] MDR Key: {record.mdr_report_key} | Event Type: {record.event_type}")
        print(f"Device: {record.brand_name or 'N/A'} (Product Code: {record.product_code or 'N/A'})")

        output = pipeline.process_record(record)
        latencies.append(output.processing_time_ms)

        total_op += len(output.operational_problems)
        total_mfg += len(output.manufacturing_issues)
        total_ae += len(output.adverse_events)
        total_other += len(output.other)

        print(f"Route: {output.execution_path.value} | Latency: {output.processing_time_ms:.2f} ms")
        
        # Display sample of cleaned narrative
        clean_snippet = output.cleaned_narrative.replace("\n", " ")[:140]
        print(f"Cleaned Narrative: \"{clean_snippet}...\"")

        if output.operational_problems:
            print("  [Operational Problems]:")
            for f in output.operational_problems:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
                print(f"    - \"{f.verbatim_span}\" (Timing: {f.temporal_timing.value}) -> IMDRF: [{codes}]")

        if output.manufacturing_issues:
            print("  [Manufacturing Issues]:")
            for f in output.manufacturing_issues:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
                print(f"    - \"{f.verbatim_span}\" (Timing: {f.temporal_timing.value}) -> IMDRF: [{codes}]")

        if output.adverse_events:
            print("  [Adverse Events]:")
            for f in output.adverse_events:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
                print(f"    - \"{f.verbatim_span}\" (Timing: {f.temporal_timing.value}) -> IMDRF/MedDRA: [{codes}]")

        if output.clinical_interventions:
            print("  [Clinical Interventions]:")
            for f in output.clinical_interventions:
                codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
                trig = f" -> Triggered by: {f.triggered_by}" if f.triggered_by else ""
                print(f"    - \"{f.verbatim_span}\"{trig} -> IMDRF: [{codes}]")

        if output.other:
            print("  [Other]:")
            for f in output.other:
                print(f"    - \"{f.verbatim_span}\"")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print("\n" + "=" * 80)
    print("REAL FDA TEST SUMMARY & PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Total Real Records Processed : {len(records)}")
    print(f"Average Pipeline Latency     : {avg_latency:.2f} ms / record")
    print(f"Estimated Throughput         : {1000.0 / avg_latency if avg_latency > 0 else 0:.0f} records / second / core")
    print(f"Operational Problems Found   : {total_op}")
    print(f"Manufacturing Issues Found   : {total_mfg}")
    print(f"Adverse Events Found         : {total_ae}")
    print(f"Other Issues Found           : {total_other}")
    print("=" * 80)


if __name__ == "__main__":
    main()
