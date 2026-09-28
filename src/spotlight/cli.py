"""
Command-line interface (CLI) for Spotlight FDA MAUDE NLP System.
Provides instant extraction, batch processing, openFDA ingestion, and benchmarking.
"""

import sys
import argparse
import json
from typing import List, Optional
from spotlight import __version__
from spotlight.pipeline import MAUDEExtractionPipeline
from spotlight.schemas import MAUDERecordInput
from spotlight.data_fetcher import MAUDEDataFetcher


def format_extraction_text(output) -> str:
    """Renders human-readable summary of extraction findings."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"SPOTLIGHT EXTRACTION SUMMARY (Latency: {output.processing_time_ms:.2f} ms | Route: {output.execution_path.value})")
    lines.append("=" * 80)

    lines.append(f"\n1. Operational Problems ({len(output.operational_problems)}):")
    for f in output.operational_problems:
        codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
        comp = f" | Component: \"{f.affected_component}\"" if f.affected_component else ""
        lines.append(f"   [{f.id}] \"{f.verbatim_span}\"{comp} | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

    lines.append(f"\n2. Manufacturing Issues ({len(output.manufacturing_issues)}):")
    for f in output.manufacturing_issues:
        codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
        lines.append(f"   [{f.id}] \"{f.verbatim_span}\" | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

    lines.append(f"\n3. Patient Adverse Events ({len(output.adverse_events)}):")
    for f in output.adverse_events:
        codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
        lines.append(f"   [{f.id}] \"{f.verbatim_span}\" | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

    lines.append(f"\n4. Clinical Interventions ({len(output.clinical_interventions)}):")
    for f in output.clinical_interventions:
        codes = ", ".join([f"{t.code}: {t.preferred_term}" for t in f.normalized_terms]) or "None"
        trig = f" -> Triggered by: {f.triggered_by}" if f.triggered_by else ""
        lines.append(f"   [{f.id}] \"{f.verbatim_span}\"{trig} | Timing: {f.temporal_timing.value} | Codes: [{codes}]")

    lines.append(f"\n5. Other Issues ({len(output.other)}):")
    for f in output.other:
        lines.append(f"   [{f.id}] \"{f.verbatim_span}\"")

    lines.append("=" * 80)
    return "\n".join(lines)


def handle_extract(args):
    pipeline = MAUDEExtractionPipeline(use_mock_slm=True)
    text = args.text
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            print("Error: No narrative text provided. Supply text as argument or pipe via stdin.", file=sys.stderr)
            sys.exit(1)

    result = pipeline.process_narrative(text)
    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(format_extraction_text(result))


def handle_batch(args):
    pipeline = MAUDEExtractionPipeline(use_mock_slm=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                records.append(MAUDERecordInput(narrative_text=item))
            elif isinstance(item, dict):
                records.append(MAUDERecordInput(**item))
    else:
        print("Error: Input JSON must be an array of narrative strings or record objects.", file=sys.stderr)
        sys.exit(1)

    results = pipeline.process_batch(records)
    out_data = [r.model_dump() for r in results]

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Successfully processed {len(results)} records. Output written to {args.output_file}")
    else:
        print(json.dumps(out_data, indent=2))


def handle_fetch(args):
    fetcher = MAUDEDataFetcher(data_dir=args.dir)
    print(f"Fetching up to {args.limit} records from openFDA API (event_type: {args.event_type or 'All'})...")
    records = fetcher.fetch_openfda_records(limit=args.limit, event_type=args.event_type, query=args.query)
    filepath = fetcher.save_records_to_json(records, args.output)
    print(f"Successfully fetched {len(records)} records. Saved to: {filepath}")


def handle_version(args):
    print(f"Spotlight FDA MAUDE NLP System v{__version__}")


def handle_mdr(args):
    from spotlight.mdr_database import MDRDatabase, MDRCatalog
    db = MDRDatabase(db_path=args.db)

    try:
        if args.mdr_action == "sync":
            years = args.years
            if years and years != "recent" and years != "all":
                try:
                    years = [int(y.strip()) for y in years.split(",")]
                except ValueError:
                    pass
            print(f"Syncing MDR flat files into {args.db} (years={years})...")
            stats = db.sync(years=years, download_dir=args.download_dir, verbose=True)
            print("\nSync completed successfully! Current database stats:")
            print(json.dumps(stats, indent=2))

        elif args.mdr_action == "stats":
            stats = db.stats()
            print(f"Database Statistics for: {args.db}")
            print(f"  Master Reports:    {stats['master_reports']:,}")
            print(f"  Device Records:    {stats['device_records']:,}")
            print(f"  Narrative Chunks:  {stats['narrative_chunks']:,}")
            if stats["top_product_codes"]:
                print("\nTop 10 Product Codes in Database:")
                for pcode, cnt in stats["top_product_codes"].items():
                    print(f"  - {pcode}: {cnt:,} reports")
            if stats["top_brand_names"]:
                print("\nTop 10 Brand Names in Database:")
                for brand, cnt in stats["top_brand_names"].items():
                    print(f"  - {brand}: {cnt:,} reports")

            ext_stats = db.get_extraction_stats()
            print(f"\nDurable Extraction Store Statistics:")
            print(f"  Total Extracted Reports: {ext_stats['total_reports_extracted']:,}")
            print(f"  Total Findings Saved:    {ext_stats['total_findings_saved']:,}")
            if ext_stats["findings_by_category"]:
                print("  Findings by Category:")
                for cat, cnt in ext_stats["findings_by_category"].items():
                    print(f"    - {cat}: {cnt:,}")
            if ext_stats["top_imdrf_codes"]:
                print("  Top IMDRF Codes:")
                for code_info in ext_stats["top_imdrf_codes"]:
                    print(f"    - [{code_info['code']}] {code_info['term']}: {code_info['count']:,}")

        elif args.mdr_action == "query":
            only_unextracted = getattr(args, "unextracted_only", False)
            records = db.query_records(
                product_code=args.product_code,
                brand_name=args.brand_name,
                event_type=args.event_type,
                query=args.query,
                only_unextracted=only_unextracted,
                limit=args.limit,
            )
            unextracted_label = " (Unextracted Delta Only)" if only_unextracted else ""
            print(f"Found {len(records)} records matching query{unextracted_label} in {args.db}:")
            for i, r in enumerate(records, 1):
                print(f"\n[{i}] MDR Key: {r.mdr_report_key} | Product: {r.brand_name} ({r.product_code}) | Event: {r.event_type}")
                preview = r.narrative_text[:180] + "..." if len(r.narrative_text) > 180 else r.narrative_text
                print(f"    Narrative: \"{preview}\"")

        elif args.mdr_action == "extract":
            skip_already_extracted = not getattr(args, "force", False)
            db.orchestrate_pipeline(
                product_code=args.product_code,
                brand_name=args.brand_name,
                event_type=args.event_type,
                query=args.query,
                limit=args.limit,
                chunk_size=args.chunk_size,
                skip_already_extracted=skip_already_extracted,
                verbose=True,
            )
    finally:
        db.close()



def main():
    parser = argparse.ArgumentParser(
        prog="spotlight",
        description="Spotlight: High-throughput NLP Extraction System for FDA MAUDE Narratives."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract clinical findings from a narrative string")
    p_extract.add_argument("text", nargs="?", default="", help="Clinical narrative text to analyze")
    p_extract.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Command: batch
    p_batch = subparsers.add_parser("batch", help="Extract findings from a JSON file of records")
    p_batch.add_argument("input_file", help="Path to input JSON file containing array of records or text")
    p_batch.add_argument("-o", "--output-file", default=None, help="Path to save output JSON")

    # Command: fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch real-world records from the official openFDA API")
    p_fetch.add_argument("--limit", type=int, default=10, help="Number of records to fetch (max 100)")
    p_fetch.add_argument("--event-type", default=None, help="Filter by event type (Malfunction, Injury, Death)")
    p_fetch.add_argument("--query", default=None, help="Custom openFDA search filter query")
    p_fetch.add_argument("--dir", default="data", help="Output directory")
    p_fetch.add_argument("-o", "--output", default="fetched_records.json", help="Output filename")

    # Command: mdr
    p_mdr = subparsers.add_parser("mdr", help="Manage direct FDA MAUDE flat files database")
    mdr_sub = p_mdr.add_subparsers(dest="mdr_action", help="MDR Database actions")

    # mdr sync
    p_sync = mdr_sub.add_parser("sync", help="Download and ingest FDA MDR flat files into SQLite")
    p_sync.add_argument("--years", default="recent", help="Years to sync: 'recent', 'all', or comma-separated e.g. '2023,2024,2025'")
    p_sync.add_argument("--db", default="data/mdr.db", help="Path to SQLite database file")
    p_sync.add_argument("--download-dir", default="data/mdr_downloads", help="Directory for temporary archive downloads")

    # mdr stats
    p_stats = mdr_sub.add_parser("stats", help="Display record counts and device statistics from database")
    p_stats.add_argument("--db", default="data/mdr.db", help="Path to SQLite database file")

    # mdr query
    p_query = mdr_sub.add_parser("query", help="Query device reports directly from local database")
    p_query.add_argument("--db", default="data/mdr.db", help="Path to SQLite database file")
    p_query.add_argument("--product-code", default=None, help="FDA 3-letter product code (e.g. GAG)")
    p_query.add_argument("--brand-name", default=None, help="Device brand name filter")
    p_query.add_argument("--event-type", default=None, help="Event type (Malfunction, Injury, Death)")
    p_query.add_argument("--query", default=None, help="Narrative text keyword search")
    p_query.add_argument("--unextracted-only", action="store_true", help="Only return delta records not yet in extraction store")
    p_query.add_argument("--limit", type=int, default=10, help="Max records to return")

    # mdr extract
    p_mdr_extract = mdr_sub.add_parser("extract", help="Orchestrate streaming extraction pipeline on MDR database")
    p_mdr_extract.add_argument("--db", default="data/mdr.db", help="Path to SQLite database file")
    p_mdr_extract.add_argument("--product-code", default=None, help="FDA 3-letter product code (e.g. GAG)")
    p_mdr_extract.add_argument("--brand-name", default=None, help="Device brand name filter")
    p_mdr_extract.add_argument("--event-type", default=None, help="Event type (Malfunction, Injury, Death)")
    p_mdr_extract.add_argument("--query", default=None, help="Narrative text keyword search")
    p_mdr_extract.add_argument("--limit", type=int, default=None, help="Max records to extract")
    p_mdr_extract.add_argument("--chunk-size", type=int, default=500, help="Batch size for streaming extraction & database commit")
    p_mdr_extract.add_argument("--force", action="store_true", help="Re-extract records even if previously extracted")

    # Command: version
    p_ver = subparsers.add_parser("version", help="Show installed Spotlight version")

    args = parser.parse_args()

    if args.command == "extract":
        handle_extract(args)
    elif args.command == "batch":
        handle_batch(args)
    elif args.command == "fetch":
        handle_fetch(args)
    elif args.command == "mdr":
        if not args.mdr_action:
            p_mdr.print_help()
        else:
            handle_mdr(args)
    elif args.command == "version":
        handle_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
