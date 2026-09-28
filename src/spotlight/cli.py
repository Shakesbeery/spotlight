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
            force = getattr(args, "force", False)
            print(f"Syncing MDR flat files into {args.db} (years={years}, force={force})...")
            stats = db.sync(years=years, download_dir=args.download_dir, force=force, verbose=True)
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


def handle_project(args):
    from spotlight.project import SpotlightProject

    if not getattr(args, "proj_action", None):
        print("Please specify a project action: list, create, stats, extract, export. Run 'spotlight project --help' for details.")
        return

    if args.proj_action == "list":
        projects = SpotlightProject.list_projects(base_dir=args.base_dir)
        if not projects:
            print(f"No projects found in '{args.base_dir}'. Create one with 'spotlight project create <name>'.")
            return
        print(f"Spotlight Projects ({len(projects)} found in '{args.base_dir}'):")
        print("-" * 80)
        for p in projects:
            print(f"  * {p['name']:<20} | Created: {p.get('created_at', '')[:10]} | Size: {p.get('size_mb', 0):.2f} MB | {p.get('description', '')}")
        print("-" * 80)

    elif args.proj_action == "create":
        p = SpotlightProject(
            name=args.name,
            base_dir=args.base_dir,
            description=args.description or "",
            raw_db=args.raw_db,
            exist_ok=True,
        )
        print(f"Project '{p.name}' ready.")
        print(f"  Directory: {p.project_dir}")
        print(f"  Raw DB:    {p.raw_db_path}")
        print(f"  Store:     {p.extractions_db_path}")
        p.close()

    elif args.proj_action == "stats":
        p = SpotlightProject.open(name=args.name, base_dir=args.base_dir)
        try:
            s = p.stats()
            print(f"Spotlight Project: {s['name']}")
            print(f"  Location:             {s['project_dir']}")
            print(f"  Description:          {s['description']}")
            print(f"  Created At:           {s['created_at']}")
            print(f"  Shared Raw DB:        {s['raw_db_path']}")
            print(f"  Extractions DB Size:  {s['extractions_db_size_mb']} MB")
            print(f"  Total Extracted Rpts: {s['total_extracted_reports']:,}")
            print(f"  Total Findings Saved: {s['total_findings_saved']:,}")
            if s["findings_by_category"]:
                print("  Findings by Category:")
                for cat, cnt in s["findings_by_category"].items():
                    print(f"    - {cat}: {cnt:,}")
            if s["top_imdrf_codes"]:
                print("  Top IMDRF Codes:")
                for code_info in s["top_imdrf_codes"]:
                    print(f"    - [{code_info['code']}] {code_info['term']}: {code_info['count']:,}")
            if s["exports"]:
                print(f"  Exported Datasets ({len(s['exports'])}):")
                for exp in s["exports"]:
                    print(f"    - {exp}")
        finally:
            p.close()

    elif args.proj_action == "extract":
        p = SpotlightProject.open(name=args.name, base_dir=args.base_dir)
        try:
            skip_already_extracted = not getattr(args, "force", False)
            p.orchestrate_pipeline(
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
            p.close()

    elif args.proj_action == "export":
        p = SpotlightProject.open(name=args.name, base_dir=args.base_dir)
        try:
            out_file = getattr(args, "output", "vigipy_dataset.csv")
            dest = p.export_vigipy_dataset(
                filename=out_file,
                product_code=args.product_code,
                brand_name=args.brand_name,
            )
            print(f"Exported Vigipy dataset to: {dest}")
        finally:
            p.close()


def handle_registry(args):
    from spotlight.device_registry import DeviceRegistryLinker
    from spotlight.mdr_database import MDRDatabase

    if not getattr(args, "reg_action", None):
        print("Please specify a registry action: stats, link, resolve. Run 'spotlight registry --help' for details.")
        return

    linker = DeviceRegistryLinker(db_path=getattr(args, "registry_db", None))
    if getattr(args, "seed", False):
        linker.seed_mock_registry()

    try:
        if args.reg_action == "stats":
            st = linker.stats()
            print(f"Device Registry Database Statistics (db={linker.db_path}):")
            print(f"  Establishment Registrations: {st['establishments']:,}")
            print(f"  Premarket Submissions:       {st['premarket_submissions']:,}")
            print(f"  AccessGUDID Records:         {st['gudid_records']:,}")
            print(f"  Device Listings:             {st['device_listings']:,}")

        elif args.reg_action == "resolve":
            res = linker.resolve_report(
                mdr_report_key=args.mdr_key or "CLI-001",
                report_number=args.report_num,
                pma_pmn_num=args.pma_pmn,
                udi_di=args.udi_di,
                brand_name=args.brand,
                product_code=args.product_code,
            )
            if getattr(args, "json", False):
                print(json.dumps(res.model_dump(), indent=2))
            else:
                print("=" * 70)
                print(f"DEVICE RESOLUTION RESULT: {res.mdr_report_key}")
                print("=" * 70)
                print(f"  Match Tier:        {res.match_tier.value}")
                print(f"  Is Affirmative:    {res.is_affirmative}")
                print(f"  Confidence Score:  {res.confidence_score:.2f}")
                if res.manufacturer:
                    print(f"  Manufacturer:      {res.manufacturer.name}")
                    if res.manufacturer.registration_number:
                        print(f"    - Reg Number:    {res.manufacturer.registration_number}")
                    if res.manufacturer.fei_number:
                        print(f"    - FEI:           {res.manufacturer.fei_number}")
                if res.device:
                    print(f"  Resolved Device:   {res.device.proprietary_name or 'N/A'}")
                    if res.device.premarket_number:
                        print(f"    - Premarket No:  {res.device.premarket_number} ({res.device.premarket_type or 'Cleared'})")
                    if res.device.udi_di:
                        print(f"    - UDI-DI:        {res.device.udi_di}")
                    if res.device.listing_number:
                        print(f"    - Listing No:    {res.device.listing_number}")
                if res.notes:
                    print(f"  Notes:             {res.notes}")
                print("=" * 70)

        elif args.reg_action == "link":
            mdr_db = MDRDatabase(db_path=args.db)
            try:
                linker.link_mdr_database(
                    mdr_db=mdr_db,
                    limit=args.limit,
                    verbose=True,
                )
            finally:
                mdr_db.close()
    finally:
        linker.close()




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
    p_sync.add_argument("--force", action="store_true", help="Force re-download and re-ingest archives even if already cached")

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

    # Command: project
    p_proj = subparsers.add_parser("project", help="Manage isolated user projects and scoped extraction spaces")
    proj_sub = p_proj.add_subparsers(dest="proj_action", help="Project actions")

    # project list
    p_plist = proj_sub.add_parser("list", help="List all project workspaces")
    p_plist.add_argument("--base-dir", default="projects", help="Base directory for projects")

    # project create
    p_pcreate = proj_sub.add_parser("create", help="Create or initialize a project workspace")
    p_pcreate.add_argument("name", help="Name of the project")
    p_pcreate.add_argument("--desc", dest="description", default="", help="Project description")
    p_pcreate.add_argument("--base-dir", default="projects", help="Base directory for projects")
    p_pcreate.add_argument("--raw-db", default="data/mdr.db", help="Path to shared raw MDR database")

    # project stats
    p_pstats = proj_sub.add_parser("stats", help="Display statistics for a project workspace")
    p_pstats.add_argument("name", help="Name of the project")
    p_pstats.add_argument("--base-dir", default="projects", help="Base directory for projects")

    # project extract
    p_pextract = proj_sub.add_parser("extract", help="Run streaming extraction scoped strictly to project workspace")
    p_pextract.add_argument("name", help="Name of the project")
    p_pextract.add_argument("--base-dir", default="projects", help="Base directory for projects")
    p_pextract.add_argument("--product-code", default=None, help="FDA 3-letter product code (e.g. GAG)")
    p_pextract.add_argument("--brand-name", default=None, help="Device brand name filter")
    p_pextract.add_argument("--event-type", default=None, help="Event type (Malfunction, Injury, Death)")
    p_pextract.add_argument("--query", default=None, help="Narrative text keyword search")
    p_pextract.add_argument("--limit", type=int, default=None, help="Max records to extract")
    p_pextract.add_argument("--chunk-size", type=int, default=500, help="Batch size for extraction & commit")
    p_pextract.add_argument("--force", action="store_true", help="Re-extract records even if previously extracted in this project")

    # project export
    p_pexport = proj_sub.add_parser("export", help="Export project findings to CSV or Parquet in exports/")
    p_pexport.add_argument("name", help="Name of the project")
    p_pexport.add_argument("--base-dir", default="projects", help="Base directory for projects")
    p_pexport.add_argument("-o", "--output", default="vigipy_dataset.csv", help="Output filename (.csv or .parquet)")
    p_pexport.add_argument("--product-code", default=None, help="Optional product code filter")
    p_pexport.add_argument("--brand-name", default=None, help="Optional brand name filter")

    # Command: registry
    p_reg = subparsers.add_parser("registry", help="Optional FDA device & manufacturer registry resolution")
    reg_sub = p_reg.add_subparsers(dest="reg_action", help="Registry actions")

    # registry stats
    p_rstats = reg_sub.add_parser("stats", help="Display record counts in reference registry database")
    p_rstats.add_argument("--registry-db", default=None, help="Path to registry SQLite database (default: in-memory)")
    p_rstats.add_argument("--seed", action="store_true", help="Pre-seed mock registry for instant offline testing")

    # registry resolve
    p_rresolve = reg_sub.add_parser("resolve", help="Test waterfall resolution for specific identifiers")
    p_rresolve.add_argument("--mdr-key", default="CLI-001", help="Target MDR Report Key")
    p_rresolve.add_argument("--report-num", default=None, help="21 CFR § 803 3-segment report number (e.g. 2183427-2024-00192)")
    p_rresolve.add_argument("--pma-pmn", default=None, help="510(k), PMA, or De Novo clearance number (e.g. K201452, P160002)")
    p_rresolve.add_argument("--udi-di", default=None, help="AccessGUDID device identifier barcode (e.g. 00884521034812)")
    p_rresolve.add_argument("--brand", default=None, help="Reported device brand/trade name")
    p_rresolve.add_argument("--product-code", default=None, help="FDA 3-letter product code (e.g. GAG)")
    p_rresolve.add_argument("--registry-db", default=None, help="Path to registry SQLite database (default: in-memory)")
    p_rresolve.add_argument("--seed", action="store_true", help="Pre-seed mock registry for instant offline testing")
    p_rresolve.add_argument("--json", action="store_true", help="Output full JSON resolution object")

    # registry link
    p_rlink = reg_sub.add_parser("link", help="Non-destructively link MDR database reports to official FDA registrations")
    p_rlink.add_argument("--db", default="data/mdr.db", help="Path to target MDR SQLite database to annotate")
    p_rlink.add_argument("--registry-db", default=None, help="Path to registry SQLite database (default: in-memory)")
    p_rlink.add_argument("--seed", action="store_true", help="Pre-seed mock registry for instant offline testing")
    p_rlink.add_argument("--limit", type=int, default=None, help="Maximum number of records to process")

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
    elif args.command == "project":
        if not args.proj_action:
            p_proj.print_help()
        else:
            handle_project(args)
    elif args.command == "registry":
        if not args.reg_action:
            p_reg.print_help()
        else:
            handle_registry(args)
    elif args.command == "version":
        handle_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
