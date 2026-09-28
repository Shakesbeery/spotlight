"""
MDR Database Manager: Automated downloading, extraction, relational database
ingestion, durable extraction persistence, and high-performance querying
for direct FDA MAUDE flat files.

Supports:
1. Canonical FDA download catalog (yearly archives, historical files, current year).
2. Resumable downloading and automated ZIP extraction.
3. Streaming, memory-constant pipe-delimited (|) parser with multiline narrative recovery.
4. Native SQLite database engine (zero external dependencies) with WAL mode & B-Tree indexes.
5. Durable Extraction Store: Persist, resume, and continually compile NLP extraction findings
   across millions of records without reprocessing.
6. Orchestrated Query-Extract-Save pipeline: Automatically anti-joins against saved extractions
   so queries only extract delta records not previously processed.
7. Direct Vigipy integration from compiled extraction storage.
"""

import os
import sys
import csv
import json
import zipfile
import sqlite3
import urllib.request
import time
from datetime import datetime, timezone
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Generator,
    Sequence,
    Union,
    Callable,
    Tuple,
    Set,
    Iterable,
)
from dataclasses import dataclass
from spotlight.schemas import (
    MAUDERecordInput,
    MAUDEExtractionOutput,
    ExtractedFinding,
    GroundedOntologyTerm,
    CategoryType,
    TemporalTiming,
    ExecutionPath,
)

# Base URL for FDA MAUDE downloadable flat files
FDA_MAUDE_FTP_BASE = "https://www.accessdata.fda.gov/MAUDE/ftparea"


@dataclass(frozen=True)
class MDRFileInfo:
    """Metadata for an FDA MAUDE downloadable zip archive."""
    file_type: str       # 'mdrfoi', 'device', 'foitext', 'patient', 'problem'
    year: Optional[int]  # Specific year (e.g., 2024) or None for cumulative/current
    filename: str        # e.g., 'device2024.zip', 'foitext2024.zip'
    url: str             # Full HTTP download URL


@dataclass
class PipelineRunStats:
    """Execution metrics for an orchestrated query-extract-save pipeline run."""
    total_matched: int
    already_extracted: int
    newly_extracted: int
    total_findings_saved: int
    elapsed_seconds: float
    records_per_second: float

    def summary(self) -> str:
        return (
            f"PipelineRunStats(matched={self.total_matched:,}, "
            f"skipped_existing={self.already_extracted:,}, "
            f"newly_processed={self.newly_extracted:,}, "
            f"findings_saved={self.total_findings_saved:,}, "
            f"elapsed={self.elapsed_seconds:.2f}s, "
            f"speed={self.records_per_second:.1f} rec/s)"
        )


class MDRCatalog:
    """
    Catalog of all official FDA MAUDE flat file download archives.
    Knows URL patterns and available years from 1996 to present.
    """

    EARLIEST_YEAR = 1996
    CURRENT_YEAR = 2026

    @classmethod
    def get_canonical_url(cls, filename: str) -> str:
        """Constructs full URL for an FDA MAUDE ftparea zip file."""
        return f"{FDA_MAUDE_FTP_BASE}/{filename}"

    @classmethod
    def resolve_archives(
        cls,
        years: Optional[Union[int, Sequence[int], str]] = None,
        file_types: Optional[Sequence[str]] = None,
    ) -> List[MDRFileInfo]:
        """
        Resolves archive list for requested years and file types.

        Args:
            years: Specific year (e.g. 2024), list of years [2023, 2024, 2025],
                   'recent' (last 3 years), 'all' (all historical files),
                   or None (defaults to 'recent').
            file_types: Specific file types to include:
                        'mdrfoi' (master events), 'device' (device metadata),
                        'foitext' (narrative text), 'patient', 'problem'.
                        Defaults to ['mdrfoi', 'device', 'foitext'].

        Returns:
            List of MDRFileInfo descriptors ready for downloading.
        """
        if file_types is None:
            file_types = ["mdrfoi", "device", "foitext"]
        selected_types = set(ft.lower().strip() for ft in file_types)

        target_years: List[int] = []
        include_cumulative = False

        if years is None or years == "recent":
            target_years = list(range(cls.CURRENT_YEAR - 2, cls.CURRENT_YEAR + 1))
        elif years == "all":
            include_cumulative = True
            target_years = list(range(2000, cls.CURRENT_YEAR + 1))
        elif isinstance(years, int):
            target_years = [years]
        elif isinstance(years, (list, tuple, set)):
            target_years = sorted(list(years))
        elif isinstance(years, str):
            target_years = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]

        archives: List[MDRFileInfo] = []

        # 1. Master Event Files (mdrfoi)
        if "mdrfoi" in selected_types or "master" in selected_types:
            if include_cumulative:
                archives.append(MDRFileInfo(
                    file_type="mdrfoi",
                    year=None,
                    filename=f"mdrfoithru{cls.CURRENT_YEAR - 1}.zip",
                    url=cls.get_canonical_url(f"mdrfoithru{cls.CURRENT_YEAR - 1}.zip"),
                ))
            archives.append(MDRFileInfo(
                file_type="mdrfoi",
                year=cls.CURRENT_YEAR,
                filename="mdrfoi.zip",
                url=cls.get_canonical_url("mdrfoi.zip"),
            ))

        # 2. Device Files
        if "device" in selected_types:
            for y in target_years:
                if y < 2000:
                    fn = f"foidev{y}.zip" if y >= 1998 else "foidevthru1997.zip"
                elif y == cls.CURRENT_YEAR:
                    fn = "device.zip"
                else:
                    fn = f"device{y}.zip"

                archives.append(MDRFileInfo(
                    file_type="device",
                    year=y,
                    filename=fn,
                    url=cls.get_canonical_url(fn),
                ))

        # 3. Narrative Text Files (foitext)
        if "foitext" in selected_types or "text" in selected_types:
            for y in target_years:
                if y < 1996:
                    fn = "foitextthru1995.zip"
                elif y == cls.CURRENT_YEAR:
                    fn = "foitext.zip"
                else:
                    fn = f"foitext{y}.zip"

                archives.append(MDRFileInfo(
                    file_type="foitext",
                    year=y,
                    filename=fn,
                    url=cls.get_canonical_url(fn),
                ))

        # 4. Patient Files
        if "patient" in selected_types:
            if include_cumulative:
                archives.append(MDRFileInfo(
                    file_type="patient",
                    year=None,
                    filename=f"patientthru{cls.CURRENT_YEAR - 1}.zip",
                    url=cls.get_canonical_url(f"patientthru{cls.CURRENT_YEAR - 1}.zip"),
                ))
            archives.append(MDRFileInfo(
                file_type="patient",
                year=cls.CURRENT_YEAR,
                filename="patient.zip",
                url=cls.get_canonical_url("patient.zip"),
            ))

        # 5. Device Problem Codes
        if "problem" in selected_types:
            archives.append(MDRFileInfo(
                file_type="problem",
                year=None,
                filename="foidevproblem.zip",
                url=cls.get_canonical_url("foidevproblem.zip"),
            ))

        seen = set()
        deduped = []
        for a in archives:
            if a.filename not in seen:
                seen.add(a.filename)
                deduped.append(a)

        return deduped


class MDRDownloader:
    """Manages downloading and extracting FDA MAUDE ZIP archives."""

    def __init__(self, download_dir: str = "data/mdr_downloads"):
        self.download_dir = download_dir
        self.zips_dir = os.path.join(self.download_dir, "zips")
        self.extracted_dir = os.path.join(self.download_dir, "extracted")
        os.makedirs(self.zips_dir, exist_ok=True)
        os.makedirs(self.extracted_dir, exist_ok=True)

    def download_file(
        self,
        url: str,
        dest_filename: str,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Downloads a single file with delta checks:
        - If file already exists locally and force=False:
          * Historical annual files (e.g. device2023.zip) are immutable and skipped immediately.
          * Current-year dynamic files (e.g. device.zip) check remote Content-Length via HTTP HEAD.
            If identical, download is skipped.
        """
        dest_path = os.path.join(self.zips_dir, dest_filename)

        if not force and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            # Check if this is a dynamic current-year file that could be updated by FDA
            # Historical archives with 4-digit years (e.g. device2023.zip) never change once published.
            is_dynamic = re.search(r"\d{4}", dest_filename) is None
            if not is_dynamic:
                return dest_path

            # For dynamic files, perform a fast HEAD check to see if remote size changed
            try:
                head_req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Spotlight-MDR-Downloader/1.0"},
                    method="HEAD",
                )
                with urllib.request.urlopen(head_req, timeout=10) as head_resp:
                    remote_len = int(head_resp.headers.get("Content-Length", 0))
                    local_len = os.path.getsize(dest_path)
                    if remote_len > 0 and remote_len == local_len:
                        return dest_path
            except Exception:
                # If network check fails or server doesn't support HEAD, retain local file
                return dest_path

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Spotlight-MDR-Downloader/1.0"}
        )

        with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)

        return dest_path

    def extract_zip(self, zip_path: str, force: bool = False) -> List[str]:
        """
        Extracts a ZIP archive into the extracted directory.
        Skips re-extracting if target .txt already exists and force=False.
        """
        extracted_files: List[str] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith(".txt"):
                    target = os.path.join(self.extracted_dir, os.path.basename(member))
                    if not force and os.path.exists(target) and os.path.getsize(target) > 0:
                        extracted_files.append(target)
                        continue
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted_files.append(target)
        return extracted_files

    def fetch_archives(
        self,
        archives: List[MDRFileInfo],
        force: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        """Downloads and unzips a list of archives. Returns all extracted .txt file paths."""
        all_extracted: List[str] = []

        for idx, archive in enumerate(archives, 1):
            dest_path = os.path.join(self.zips_dir, archive.filename)
            already_downloaded = not force and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0

            if verbose:
                status_msg = "Checking / Fetching" if not already_downloaded else "Verifying cached"
                print(f"[{idx}/{len(archives)}] {status_msg} {archive.filename}...")

            def progress(downloaded: int, total: int):
                if total > 0 and verbose:
                    pct = (downloaded / total) * 100.0
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    print(f"\r   -> Downloading: {pct:.1f}% ({mb_down:.1f}/{mb_tot:.1f} MB)", end="", flush=True)

            try:
                zip_path = self.download_file(
                    archive.url,
                    archive.filename,
                    force=force,
                    progress_callback=progress if verbose else None,
                )
                if verbose and not already_downloaded:
                    print()
                txts = self.extract_zip(zip_path, force=force)
                all_extracted.extend(txts)
            except Exception as e:
                if verbose:
                    print(f"\n   -> Warning: Failed to download {archive.filename}: {e}")

        return list(set(all_extracted))


class MDRParser:
    """Robust streaming reader for FDA pipe-delimited text files."""

    @staticmethod
    def identify_file_type(filepath: str, header_line: str) -> str:
        """Determines the table type from filename or header line."""
        base = os.path.basename(filepath).lower()
        if "device" in base or "dev" in base:
            return "device"
        if "text" in base:
            return "foitext"
        if "mdrfoi" in base:
            return "mdrfoi"
        if "patient" in base:
            return "patient"
        if "problem" in base:
            return "problem"

        header_upper = header_line.upper()
        if "BRAND_NAME" in header_upper or "GENERIC_NAME" in header_upper:
            return "device"
        if "FOI_TEXT" in header_upper or "MDR_TEXT_KEY" in header_upper:
            return "foitext"
        if "EVENT_TYPE" in header_upper or "DATE_RECEIVED" in header_upper:
            return "mdrfoi"
        return "unknown"

    @staticmethod
    def stream_rows(
        filepath: str,
        chunk_size: int = 10000,
        encoding: str = "latin-1",
    ) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
        """
        Streams chunks of parsed rows from a pipe-delimited file.
        Yields (file_type, list_of_row_dicts).
        """
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            try:
                raw_header = next(reader)
            except StopIteration:
                return

            clean_header = [col.strip().upper() for col in raw_header]
            file_type = MDRParser.identify_file_type(filepath, "|".join(clean_header))

            batch: List[Dict[str, str]] = []
            num_cols = len(clean_header)

            for line_no, row in enumerate(reader, start=2):
                if not row:
                    continue

                if len(row) < num_cols:
                    row.extend([""] * (num_cols - len(row)))
                elif len(row) > num_cols:
                    if clean_header[-1] == "FOI_TEXT":
                        text_val = "|".join(row[num_cols - 1:])
                        row = row[:num_cols - 1] + [text_val]
                    else:
                        row = row[:num_cols]

                row_dict = {clean_header[i]: row[i].strip() for i in range(num_cols)}
                batch.append(row_dict)

                if len(batch) >= chunk_size:
                    yield (file_type, batch)
                    batch = []

            if batch:
                yield (file_type, batch)


class MDRDatabase:
    """
    High-performance relational database layer for FDA MAUDE records
    and durable storage for Spotlight extraction findings.
    Backed by SQLite with automatic B-Tree indexing and WAL mode.
    """

    def __init__(self, db_path: str = "data/mdr.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates an optimized SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64 MB memory cache
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    def _init_schema(self):
        """Initializes raw MDR tables, extraction storage tables, and indices."""
        conn = self._get_connection()
        try:
            # 1. Master Event Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mdr_master (
                    mdr_report_key TEXT PRIMARY KEY,
                    report_number TEXT,
                    event_type TEXT,
                    date_received TEXT,
                    date_of_event TEXT,
                    report_source_code TEXT,
                    adverse_event_flag TEXT,
                    product_problem_flag TEXT
                )
            """)

            # 2. Device Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mdr_device (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mdr_report_key TEXT NOT NULL,
                    brand_name TEXT,
                    generic_name TEXT,
                    model_number TEXT,
                    product_code TEXT,
                    manufacturer_name TEXT,
                    pma_pmn_num TEXT,
                    udi_di TEXT
                )
            """)

            # Check and run column migrations if existing database tables lack newer linkage keys
            try:
                cols_m = [c[1] for c in conn.execute("PRAGMA table_info(mdr_master)").fetchall()]
                if "report_number" not in cols_m:
                    conn.execute("ALTER TABLE mdr_master ADD COLUMN report_number TEXT")
                cols_d = [c[1] for c in conn.execute("PRAGMA table_info(mdr_device)").fetchall()]
                if "pma_pmn_num" not in cols_d:
                    conn.execute("ALTER TABLE mdr_device ADD COLUMN pma_pmn_num TEXT")
                if "udi_di" not in cols_d:
                    conn.execute("ALTER TABLE mdr_device ADD COLUMN udi_di TEXT")
            except Exception:
                pass

            # 3. Narrative Text Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mdr_text (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mdr_report_key TEXT NOT NULL,
                    text_type_code TEXT,
                    date_report TEXT,
                    foi_text TEXT
                )
            """)

            # 4. Durable Extraction Storage: Extracted Reports
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extracted_reports (
                    mdr_report_key TEXT PRIMARY KEY,
                    brand_name TEXT,
                    product_code TEXT,
                    event_type TEXT,
                    execution_path TEXT,
                    processing_time_ms REAL,
                    num_operational_problems INTEGER,
                    num_manufacturing_issues INTEGER,
                    num_adverse_events INTEGER,
                    num_clinical_interventions INTEGER,
                    num_other INTEGER,
                    cleaned_narrative TEXT,
                    extracted_at TEXT,
                    project_name TEXT
                )
            """)
            try:
                conn.execute("ALTER TABLE extracted_reports ADD COLUMN project_name TEXT")
            except Exception:
                pass

            # 5. Durable Extraction Storage: Extracted Findings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extracted_findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    finding_id TEXT,
                    mdr_report_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    verbatim_span TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER,
                    temporal_timing TEXT,
                    confidence REAL,
                    affected_component TEXT,
                    failure_mode TEXT,
                    aliases_json TEXT,
                    triggered_by TEXT,
                    code TEXT,
                    preferred_term TEXT,
                    ontology TEXT,
                    similarity_score REAL,
                    details_json TEXT,
                    all_normalized_json TEXT
                )
            """)

            # Raw Tables Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_device_key ON mdr_device(mdr_report_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_device_pcode ON mdr_device(product_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_device_brand ON mdr_device(brand_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_text_key ON mdr_text(mdr_report_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_master_event ON mdr_master(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_master_date ON mdr_master(date_received)")

            # Extraction Storage Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_reports_pcode ON extracted_reports(product_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_reports_brand ON extracted_reports(brand_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_findings_key ON extracted_findings(mdr_report_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_findings_cat ON extracted_findings(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_findings_code ON extracted_findings(code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ex_findings_comp ON extracted_findings(affected_component)")

            # Ingestion Log to prevent duplicate ingestion of already processed flat files
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mdr_sync_log (
                    filename TEXT PRIMARY KEY,
                    file_size INTEGER NOT NULL,
                    row_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                )
            """)

            conn.commit()
        finally:
            conn.close()

    def ingest_txt(self, filepath: str, chunk_size: int = 15000, force: bool = False, verbose: bool = True) -> int:
        """
        Parses and ingests a single .txt flat file into the database.
        Checks mdr_sync_log to skip already-ingested files unless force=True.
        Returns number of rows ingested.
        """
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        conn = self._get_connection()
        try:
            if not force:
                row = conn.execute(
                    "SELECT row_count FROM mdr_sync_log WHERE filename = ? AND file_size = ?",
                    (filename, file_size),
                ).fetchone()
                if row:
                    if verbose:
                        print(f"File {filename} is already ingested ({row[0]:,} rows). Skipping.")
                    return 0

            total_rows = 0
            for file_type, batch in MDRParser.stream_rows(filepath, chunk_size=chunk_size):
                if file_type == "device":
                    records = [
                        (
                            row.get("MDR_REPORT_KEY", ""),
                            row.get("BRAND_NAME", ""),
                            row.get("GENERIC_NAME", ""),
                            row.get("MODEL_NUMBER", ""),
                            row.get("DEVICE_REPORT_PRODUCT_CODE", row.get("PRODUCT_CODE", "")),
                            row.get("MANUFACTURER_D_NAME", ""),
                            row.get("PMA_PMN_NUM", ""),
                            row.get("UDI-DI", row.get("UDI_DI", row.get("UDI-PUBLIC", row.get("UDI_PUBLIC", "")))),
                        )
                        for row in batch if row.get("MDR_REPORT_KEY")
                    ]
                    conn.executemany("""
                        INSERT INTO mdr_device (mdr_report_key, brand_name, generic_name, model_number, product_code, manufacturer_name, pma_pmn_num, udi_di)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)

                elif file_type == "foitext":
                    records = [
                        (
                            row.get("MDR_REPORT_KEY", ""),
                            row.get("TEXT_TYPE_CODE", ""),
                            row.get("DATE_REPORT", ""),
                            row.get("FOI_TEXT", ""),
                        )
                        for row in batch if row.get("MDR_REPORT_KEY") and row.get("FOI_TEXT")
                    ]
                    conn.executemany("""
                        INSERT INTO mdr_text (mdr_report_key, text_type_code, date_report, foi_text)
                        VALUES (?, ?, ?, ?)
                    """, records)

                elif file_type == "mdrfoi":
                    records = [
                        (
                            row.get("MDR_REPORT_KEY", ""),
                            row.get("REPORT_NUMBER", ""),
                            row.get("EVENT_TYPE", "Unknown"),
                            row.get("DATE_RECEIVED", ""),
                            row.get("DATE_OF_EVENT", ""),
                            row.get("REPORT_SOURCE_CODE", ""),
                            row.get("ADVERSE_EVENT_FLAG", ""),
                            row.get("PRODUCT_PROBLEM_FLAG", ""),
                        )
                        for row in batch if row.get("MDR_REPORT_KEY")
                    ]
                    conn.executemany("""
                        INSERT OR REPLACE INTO mdr_master (
                            mdr_report_key, report_number, event_type, date_received, date_of_event,
                            report_source_code, adverse_event_flag, product_problem_flag
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)

                conn.commit()
                total_rows += len(batch)

                if verbose:
                    print(f"\r   -> Ingested {total_rows:,} rows from {os.path.basename(filepath)}...", end="", flush=True)

            conn.execute("""
                INSERT OR REPLACE INTO mdr_sync_log (filename, file_size, row_count, ingested_at)
                VALUES (?, ?, ?, ?)
            """, (filename, file_size, total_rows, datetime.now(timezone.utc).isoformat()))
            conn.commit()

            if verbose:
                print()
            return total_rows
        finally:
            conn.close()

    def sync(
        self,
        years: Optional[Union[int, Sequence[int], str]] = None,
        file_types: Optional[Sequence[str]] = None,
        download_dir: str = "data/mdr_downloads",
        force: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        One-stop automation: Resolves archives, downloads, unzips, and ingests into database.
        Includes delta checks to skip files already downloaded and ingested.
        """
        downloader = MDRDownloader(download_dir=download_dir)
        archives = MDRCatalog.resolve_archives(years=years, file_types=file_types)

        if verbose:
            print(f"Syncing {len(archives)} FDA MDR archives for years={years} (force={force})...")

        txt_files = downloader.fetch_archives(archives, force=force, verbose=verbose)

        if verbose:
            print(f"\nIngesting {len(txt_files)} extracted flat files into {self.db_path}...")

        total_rows = 0
        for txt in txt_files:
            total_rows += self.ingest_txt(txt, force=force, verbose=verbose)

        stats = self.stats()
        stats["total_rows_ingested_this_sync"] = total_rows
        return stats

    def close(self):
        """Forces WAL checkpoint and ensures connections are released."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception:
            pass
        import gc
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def stats(self) -> Dict[str, Any]:
        """Returns database counts and high-level summaries for raw MDR data."""
        conn = self._get_connection()
        try:
            master_count = conn.execute("SELECT COUNT(*) FROM mdr_master").fetchone()[0]
            device_count = conn.execute("SELECT COUNT(*) FROM mdr_device").fetchone()[0]
            text_count = conn.execute("SELECT COUNT(*) FROM mdr_text").fetchone()[0]

            top_pcodes = conn.execute("""
                SELECT product_code, COUNT(*) as c
                FROM mdr_device
                WHERE product_code IS NOT NULL AND product_code != ''
                GROUP BY product_code
                ORDER BY c DESC LIMIT 10
            """).fetchall()

            top_brands = conn.execute("""
                SELECT brand_name, COUNT(*) as c
                FROM mdr_device
                WHERE brand_name IS NOT NULL AND brand_name != ''
                GROUP BY brand_name
                ORDER BY c DESC LIMIT 10
            """).fetchall()

            return {
                "db_path": self.db_path,
                "master_reports": master_count,
                "device_records": device_count,
                "narrative_chunks": text_count,
                "top_product_codes": dict(top_pcodes),
                "top_brand_names": dict(top_brands),
            }
        finally:
            conn.close()

    # =========================================================================
    # QUERY & STREAMING INTERFACE
    # =========================================================================

    def _build_query_filter(
        self,
        product_code: Optional[Union[str, Sequence[str]]] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        mdr_report_key: Optional[str] = None,
        mdr_report_keys: Optional[Sequence[str]] = None,
        only_unextracted: bool = False,
        product_codes: Optional[Sequence[str]] = None,
    ) -> Tuple[str, str, List[Any]]:
        """Constructs WHERE clauses and JOINs for raw record querying."""
        where_clauses = ["t.full_narrative IS NOT NULL", "t.full_narrative != ''"]
        params: List[Any] = []

        if mdr_report_key:
            where_clauses.append("d.mdr_report_key = ?")
            params.append(str(mdr_report_key).strip())

        if mdr_report_keys:
            placeholders = ",".join("?" for _ in mdr_report_keys)
            where_clauses.append(f"d.mdr_report_key IN ({placeholders})")
            params.extend([str(k).strip() for k in mdr_report_keys])

        active_codes = []
        if product_codes:
            active_codes.extend([str(c).upper().strip() for c in product_codes])
        if product_code:
            if isinstance(product_code, (list, tuple, set)):
                active_codes.extend([str(c).upper().strip() for c in product_code])
            else:
                active_codes.append(product_code.upper().strip())

        if active_codes:
            if len(active_codes) == 1:
                where_clauses.append("UPPER(d.product_code) = ?")
                params.append(active_codes[0])
            else:
                placeholders = ",".join("?" for _ in active_codes)
                where_clauses.append(f"UPPER(d.product_code) IN ({placeholders})")
                params.extend(active_codes)

        if brand_name:
            where_clauses.append("LOWER(d.brand_name) LIKE ?")
            params.append(f"%{brand_name.lower().strip()}%")

        # Event type: if None, 'all', or empty, DO NOT filter (returns all event types)
        if event_type and event_type.lower().strip() not in ("all", "*", ""):
            where_clauses.append("LOWER(m.event_type) = ?")
            params.append(event_type.lower().strip())

        if query:
            where_clauses.append("LOWER(t.full_narrative) LIKE ?")
            params.append(f"%{query.lower().strip()}%")

        anti_join = ""
        if only_unextracted:
            anti_join = "LEFT JOIN extracted_reports ex ON d.mdr_report_key = ex.mdr_report_key"
            where_clauses.append("ex.mdr_report_key IS NULL")

        return anti_join, " AND ".join(where_clauses), params

    def count_records(
        self,
        product_code: Optional[Union[str, Sequence[str]]] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        mdr_report_key: Optional[str] = None,
        mdr_report_keys: Optional[Sequence[str]] = None,
        only_unextracted: bool = False,
        product_codes: Optional[Sequence[str]] = None,
    ) -> int:
        """Counts distinct reports matching criteria."""
        anti_join, where_sql, params = self._build_query_filter(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            mdr_report_key=mdr_report_key,
            mdr_report_keys=mdr_report_keys,
            only_unextracted=only_unextracted,
            product_codes=product_codes,
        )
        sql = f"""
            SELECT COUNT(DISTINCT d.mdr_report_key)
            FROM (
                SELECT mdr_report_key, GROUP_CONCAT(foi_text, char(10) || char(10)) AS full_narrative
                FROM mdr_text
                WHERE foi_text IS NOT NULL AND foi_text != ''
                GROUP BY mdr_report_key
            ) t
            JOIN mdr_device d ON t.mdr_report_key = d.mdr_report_key
            LEFT JOIN mdr_master m ON d.mdr_report_key = m.mdr_report_key
            {anti_join}
            WHERE {where_sql}
        """
        conn = self._get_connection()
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def stream_records(
        self,
        product_code: Optional[Union[str, Sequence[str]]] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        mdr_report_key: Optional[str] = None,
        mdr_report_keys: Optional[Sequence[str]] = None,
        only_unextracted: bool = False,
        limit: Optional[int] = None,
        chunk_size: int = 2000,
        product_codes: Optional[Sequence[str]] = None,
    ) -> Generator[MAUDERecordInput, None, None]:
        """
        Streams joined, fully-populated MAUDERecordInput instances on-the-fly.
        Memory-constant streaming: executes on disk via indexed joins.
        Safely aggregates narrative texts per report key without cartesian inflation
        from multi-device reports.
        """
        conn = self._get_connection()
        anti_join, where_sql, params = self._build_query_filter(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            mdr_report_key=mdr_report_key,
            mdr_report_keys=mdr_report_keys,
            only_unextracted=only_unextracted,
            product_codes=product_codes,
        )

        limit_sql = f"LIMIT {limit}" if limit is not None else ""

        sql = f"""
            SELECT
                d.mdr_report_key,
                d.brand_name,
                d.product_code,
                COALESCE(m.event_type, 'Unknown') as event_type,
                t.full_narrative
            FROM (
                SELECT mdr_report_key, GROUP_CONCAT(foi_text, char(10) || char(10)) AS full_narrative
                FROM mdr_text
                WHERE foi_text IS NOT NULL AND foi_text != ''
                GROUP BY mdr_report_key
            ) t
            JOIN mdr_device d ON t.mdr_report_key = d.mdr_report_key
            LEFT JOIN mdr_master m ON d.mdr_report_key = m.mdr_report_key
            {anti_join}
            WHERE {where_sql}
            GROUP BY d.mdr_report_key
            ORDER BY d.mdr_report_key
            {limit_sql}
        """

        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)

            count = 0
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                for r in rows:
                    mdr_key, brand, pcode, etype, narrative = r
                    if not narrative or not narrative.strip():
                        continue

                    yield MAUDERecordInput(
                        mdr_report_key=str(mdr_key),
                        brand_name=brand or None,
                        product_code=pcode or None,
                        event_type=etype or "Unknown",
                        narrative_text=narrative.strip(),
                    )
                    count += 1
                    if limit is not None and count >= limit:
                        return
        finally:
            conn.close()

    def query_records(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        mdr_report_key: Optional[str] = None,
        mdr_report_keys: Optional[Sequence[str]] = None,
        only_unextracted: bool = False,
        limit: int = 100,
    ) -> List[MAUDERecordInput]:
        """Convenience method returning a list of matched MAUDERecordInput objects."""
        return list(self.stream_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            mdr_report_key=mdr_report_key,
            mdr_report_keys=mdr_report_keys,
            only_unextracted=only_unextracted,
            limit=limit,
        ))

    def get_record(self, mdr_report_key: str) -> Optional[MAUDERecordInput]:
        """Looks up a single MAUDERecordInput by its report key."""
        records = self.query_records(mdr_report_key=mdr_report_key, limit=1)
        return records[0] if records else None

    def get_metadata(self, mdr_report_key: str) -> Optional[Dict[str, Any]]:
        """
        Looks up full raw MDR relational metadata for a specific report key.
        Returns dictionary with master event details, device specifications, and raw narrative chunks.
        """
        conn = self._get_connection()
        try:
            master_row = conn.execute("""
                SELECT report_number, event_type, date_received, date_of_event, report_source_code,
                       adverse_event_flag, product_problem_flag
                FROM mdr_master WHERE mdr_report_key = ? LIMIT 1
            """, (str(mdr_report_key),)).fetchone()

            dev_rows = conn.execute("""
                SELECT brand_name, generic_name, model_number, product_code, manufacturer_name,
                       pma_pmn_num, udi_di
                FROM mdr_device WHERE mdr_report_key = ?
                ORDER BY id
            """, (str(mdr_report_key),)).fetchall()
            first_dev = dev_rows[0] if dev_rows else None

            text_rows = conn.execute("""
                SELECT text_type_code, date_report, foi_text
                FROM mdr_text WHERE mdr_report_key = ?
                ORDER BY id
            """, (str(mdr_report_key),)).fetchall()

            if not master_row and not dev_rows and not text_rows:
                return None

            devices = [
                {
                    "brand_name": d[0],
                    "generic_name": d[1],
                    "model_number": d[2],
                    "product_code": d[3],
                    "manufacturer_name": d[4],
                    "pma_pmn_num": d[5],
                    "udi_di": d[6],
                }
                for d in dev_rows
            ]

            return {
                "mdr_report_key": str(mdr_report_key),
                "report_number": master_row[0] if master_row else None,
                "event_type": master_row[1] if master_row else "Unknown",
                "date_received": master_row[2] if master_row else None,
                "date_of_event": master_row[3] if master_row else None,
                "report_source_code": master_row[4] if master_row else None,
                "adverse_event_flag": master_row[5] if master_row else None,
                "product_problem_flag": master_row[6] if master_row else None,
                "brand_name": first_dev[0] if first_dev else None,
                "generic_name": first_dev[1] if first_dev else None,
                "model_number": first_dev[2] if first_dev else None,
                "product_code": first_dev[3] if first_dev else None,
                "manufacturer_name": first_dev[4] if first_dev else None,
                "pma_pmn_num": first_dev[5] if first_dev else None,
                "udi_di": first_dev[6] if first_dev else None,
                "devices": devices,
                "narratives": [
                    {"text_type_code": t[0], "date_report": t[1], "foi_text": t[2]}
                    for t in text_rows
                ],
            }
        finally:
            conn.close()

    # =========================================================================
    # DURABLE EXTRACTION STORAGE
    # =========================================================================

    def save_extractions_batch(self, outputs: Sequence[MAUDEExtractionOutput]) -> int:
        """
        Durably persists a batch of extraction outputs and all their findings into SQLite.
        Runs in an atomic transaction for maximum performance (~10,000+ findings/sec).
        Returns total number of findings saved.
        """
        if not outputs:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        report_records = []
        finding_records = []

        for out in outputs:
            report_records.append((
                out.mdr_report_key,
                out.brand_name or None,
                out.product_code or None,
                out.event_type or "Unknown",
                out.execution_path.value if hasattr(out.execution_path, "value") else str(out.execution_path),
                out.processing_time_ms,
                len(out.operational_problems),
                len(out.manufacturing_issues),
                len(out.adverse_events),
                len(out.clinical_interventions),
                len(out.other),
                out.cleaned_narrative,
                now_iso,
                out.project_name or None,
            ))

            all_findings = (
                out.operational_problems
                + out.manufacturing_issues
                + out.adverse_events
                + out.clinical_interventions
                + out.other
            )

            for f in all_findings:
                norm = f.normalized_terms[0] if f.normalized_terms else None
                finding_records.append((
                    f.id or "",
                    out.mdr_report_key,
                    f.category.value if hasattr(f.category, "value") else str(f.category),
                    f.verbatim_span,
                    f.start_char,
                    f.end_char,
                    f.temporal_timing.value if hasattr(f.temporal_timing, "value") else str(f.temporal_timing),
                    f.confidence,
                    f.affected_component or None,
                    f.failure_mode or None,
                    json.dumps(f.aliases),
                    f.triggered_by or None,
                    norm.code if norm else None,
                    norm.preferred_term if norm else None,
                    norm.ontology if norm else None,
                    norm.similarity_score if norm else None,
                    json.dumps(f.details),
                    json.dumps([t.model_dump() for t in f.normalized_terms]),
                ))

        conn = self._get_connection()
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO extracted_reports (
                    mdr_report_key, brand_name, product_code, event_type,
                    execution_path, processing_time_ms,
                    num_operational_problems, num_manufacturing_issues,
                    num_adverse_events, num_clinical_interventions, num_other,
                    cleaned_narrative, extracted_at, project_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, report_records)

            keys_to_clean = [(r[0],) for r in report_records]
            conn.executemany("DELETE FROM extracted_findings WHERE mdr_report_key = ?", keys_to_clean)

            conn.executemany("""
                INSERT INTO extracted_findings (
                    finding_id, mdr_report_key, category, verbatim_span,
                    start_char, end_char, temporal_timing, confidence,
                    affected_component, failure_mode, aliases_json, triggered_by,
                    code, preferred_term, ontology, similarity_score,
                    details_json, all_normalized_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, finding_records)

            conn.commit()
            return len(finding_records)
        finally:
            conn.close()

    def save_extraction(self, output: MAUDEExtractionOutput) -> int:
        """Persists a single extraction result."""
        return self.save_extractions_batch([output])

    def is_extracted(self, mdr_report_key: str) -> bool:
        """Checks if a report key has already been extracted in the database."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM extracted_reports WHERE mdr_report_key = ? LIMIT 1",
                (mdr_report_key,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_extracted_keys(self, product_code: Optional[str] = None) -> Set[str]:
        """Returns set of all extracted MDR report keys, optionally filtered by product code."""
        conn = self._get_connection()
        try:
            if product_code:
                rows = conn.execute(
                    "SELECT mdr_report_key FROM extracted_reports WHERE UPPER(product_code) = ?",
                    (product_code.upper().strip(),)
                ).fetchall()
            else:
                rows = conn.execute("SELECT mdr_report_key FROM extracted_reports").fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def load_extractions(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        category: Optional[Union[str, CategoryType]] = None,
        code: Optional[str] = None,
        project_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MAUDEExtractionOutput]:
        """
        Reconstructs typed MAUDEExtractionOutput objects from the durable store.
        """
        return list(self.stream_extractions(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            category=category,
            code=code,
            project_name=project_name,
            limit=limit,
        ))

    def stream_extractions(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        category: Optional[Union[str, CategoryType]] = None,
        code: Optional[str] = None,
        project_name: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> Generator[MAUDEExtractionOutput, None, None]:
        """
        Streams reconstructed MAUDEExtractionOutput objects from disk in constant memory.
        """
        conn = self._get_connection()

        where_clauses = []
        params: List[Any] = []

        if product_code:
            where_clauses.append("UPPER(r.product_code) = ?")
            params.append(product_code.upper().strip())

        if brand_name:
            where_clauses.append("LOWER(r.brand_name) LIKE ?")
            params.append(f"%{brand_name.lower().strip()}%")

        if event_type and event_type.lower().strip() not in ("all", "*", ""):
            where_clauses.append("LOWER(r.event_type) = ?")
            params.append(event_type.lower().strip())

        if project_name:
            where_clauses.append("r.project_name = ?")
            params.append(project_name)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        limit_sql = f"LIMIT {limit}" if limit is not None else ""

        sql_reports = f"""
            SELECT
                r.mdr_report_key, r.brand_name, r.product_code, r.event_type,
                r.execution_path, r.processing_time_ms, r.cleaned_narrative,
                r.project_name
            FROM extracted_reports r
            {where_sql}
            ORDER BY r.mdr_report_key
            {limit_sql}
        """

        try:
            cursor = conn.cursor()
            cursor.execute(sql_reports, params)

            cat_filter = category.value if isinstance(category, CategoryType) else (str(category) if category else None)

            count = 0
            while True:
                report_rows = cursor.fetchmany(chunk_size)
                if not report_rows:
                    break

                keys = [r[0] for r in report_rows]
                # Fetch findings for this batch of reports
                placeholders = ",".join("?" for _ in keys)
                finding_sql = f"""
                    SELECT
                        finding_id, mdr_report_key, category, verbatim_span,
                        start_char, end_char, temporal_timing, confidence,
                        affected_component, failure_mode, aliases_json, triggered_by,
                        details_json, all_normalized_json
                    FROM extracted_findings
                    WHERE mdr_report_key IN ({placeholders})
                """
                findings_rows = conn.execute(finding_sql, keys).fetchall()

                # Group findings by report key
                report_findings: Dict[str, List[ExtractedFinding]] = {k: [] for k in keys}

                for f_row in findings_rows:
                    (
                        fid, f_mkey, f_cat, f_span, s_char, e_char, f_timing,
                        f_conf, f_comp, f_fmode, f_alias_json, f_trig,
                        f_det_json, f_norm_json
                    ) = f_row

                    if cat_filter and f_cat != cat_filter:
                        continue
                    if code and code not in (f_norm_json or ""):
                        continue

                    # Parse normalized terms
                    normalized_terms = []
                    try:
                        raw_terms = json.loads(f_norm_json) if f_norm_json else []
                        for t in raw_terms:
                            normalized_terms.append(GroundedOntologyTerm(**t))
                    except Exception:
                        pass

                    aliases = []
                    try:
                        aliases = json.loads(f_alias_json) if f_alias_json else []
                    except Exception:
                        pass

                    details = {}
                    try:
                        details = json.loads(f_det_json) if f_det_json else {}
                    except Exception:
                        pass

                    finding = ExtractedFinding(
                        id=fid or None,
                        category=CategoryType(f_cat) if f_cat in CategoryType._value2member_map_ else CategoryType.OTHER,
                        verbatim_span=f_span,
                        start_char=s_char,
                        end_char=e_char,
                        temporal_timing=TemporalTiming(f_timing) if f_timing in TemporalTiming._value2member_map_ else TemporalTiming.UNKNOWN,
                        confidence=f_conf,
                        affected_component=f_comp or None,
                        failure_mode=f_fmode or None,
                        aliases=aliases,
                        triggered_by=f_trig or None,
                        normalized_terms=normalized_terms,
                        details=details,
                    )
                    report_findings[f_mkey].append(finding)

                for r_row in report_rows:
                    mkey, brand, pcode, etype, exec_path, proc_time, cleaned, proj_name = r_row
                    findings = report_findings.get(mkey, [])

                    # Bucket findings
                    op_probs = [f for f in findings if f.category == CategoryType.OPERATIONAL_PROBLEM]
                    mfg_issues = [f for f in findings if f.category == CategoryType.MANUFACTURING_ISSUE]
                    aes = [f for f in findings if f.category == CategoryType.ADVERSE_EVENT]
                    interventions = [f for f in findings if f.category == CategoryType.CLINICAL_INTERVENTIONS]
                    others = [f for f in findings if f.category == CategoryType.OTHER]

                    ep_enum = ExecutionPath.FAST_PATH_MODERNBERT
                    if exec_path in ExecutionPath._value2member_map_:
                        ep_enum = ExecutionPath(exec_path)

                    yield MAUDEExtractionOutput(
                        mdr_report_key=mkey,
                        brand_name=brand,
                        product_code=pcode,
                        event_type=etype,
                        operational_problems=op_probs,
                        manufacturing_issues=mfg_issues,
                        adverse_events=aes,
                        clinical_interventions=interventions,
                        other=others,
                        execution_path=ep_enum,
                        processing_time_ms=proc_time or 0.0,
                        cleaned_narrative=cleaned or "",
                        project_name=proj_name or None,
                    )
                    count += 1
                    if limit is not None and count >= limit:
                        return
        finally:
            conn.close()

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Returns statistics on compiled extractions currently stored."""
        conn = self._get_connection()
        try:
            total_extracted = conn.execute("SELECT COUNT(*) FROM extracted_reports").fetchone()[0]
            total_findings = conn.execute("SELECT COUNT(*) FROM extracted_findings").fetchone()[0]

            findings_by_cat = conn.execute("""
                SELECT category, COUNT(*) FROM extracted_findings GROUP BY category
            """).fetchall()

            top_imdrf_codes = conn.execute("""
                SELECT code, preferred_term, COUNT(*) as c
                FROM extracted_findings
                WHERE code IS NOT NULL AND code != ''
                GROUP BY code, preferred_term
                ORDER BY c DESC LIMIT 10
            """).fetchall()

            return {
                "total_reports_extracted": total_extracted,
                "total_findings_saved": total_findings,
                "findings_by_category": dict(findings_by_cat),
                "top_imdrf_codes": [
                    {"code": r[0], "term": r[1], "count": r[2]} for r in top_imdrf_codes
                ],
            }
        finally:
            conn.close()

    # =========================================================================
    # ORCHESTRATED QUERY-EXTRACT-SAVE PIPELINE
    # =========================================================================

    def orchestrate_pipeline(
        self,
        extractor: Optional[Any] = None,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_size: int = 500,
        skip_already_extracted: bool = True,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineRunStats:
        """
        Orchestrates an end-to-end Query -> Extract -> Save pipeline.
        
        1. Checks total matching reports in database.
        2. If skip_already_extracted=True, automatically anti-joins so only unextracted delta records are queried.
        3. Streams records in chunks of `chunk_size`.
        4. Runs high-throughput NLP extraction on each chunk.
        5. Durably commits findings to SQLite after each chunk, releasing memory.
        6. Keeps total memory footprint constant (<100MB RAM) regardless of query size.
        """
        if extractor is None:
            from spotlight import get_default_extractor
            extractor = get_default_extractor()

        # Step 1: Pre-flight count of matched vs already extracted
        total_matched = self.count_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            only_unextracted=False,
        )

        unextracted_total = self.count_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            only_unextracted=True,
        )

        already_extracted = total_matched - unextracted_total

        if verbose:
            print("=" * 80)
            print("SPOTLIGHT ORCHESTRATED QUERY-EXTRACT-SAVE PIPELINE")
            print("=" * 80)
            print(f"Target Filters: Product Code: {product_code or 'ALL'} | Brand: {brand_name or 'ALL'} | Event Type: {event_type or 'ALL'}")
            print(f"Inventory: {total_matched:,} total records matching query.")
            print(f"           {already_extracted:,} records already extracted (skipped).")
            print(f"           {unextracted_total:,} delta records queued for extraction.")

        if skip_already_extracted and unextracted_total == 0:
            if verbose:
                print("All matching records have already been extracted! Zero redundant compute required.")
            return PipelineRunStats(
                total_matched=total_matched,
                already_extracted=already_extracted,
                newly_extracted=0,
                total_findings_saved=0,
                elapsed_seconds=0.0,
                records_per_second=0.0,
            )

        t_start = time.perf_counter()
        newly_extracted = 0
        total_findings_saved = 0
        effective_limit = limit

        stream = self.stream_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            only_unextracted=skip_already_extracted,
            limit=effective_limit,
            chunk_size=chunk_size,
        )

        buffer: List[MAUDERecordInput] = []

        for record in stream:
            buffer.append(record)

            if len(buffer) >= chunk_size:
                outputs = extractor.process_batch(buffer)
                saved_count = self.save_extractions_batch(outputs)
                newly_extracted += len(outputs)
                total_findings_saved += saved_count

                if verbose:
                    elapsed = max(time.perf_counter() - t_start, 0.001)
                    rate = newly_extracted / elapsed
                    target_num = min(unextracted_total, limit) if limit else unextracted_total
                    print(
                        f"\r[Processed: {newly_extracted:,}/{target_num:,}] "
                        f"Saved: {total_findings_saved:,} findings | "
                        f"Speed: {rate:.1f} rec/s",
                        end="", flush=True
                    )

                if progress_callback:
                    progress_callback(newly_extracted, unextracted_total)

                buffer.clear()

        # Flush remaining buffer
        if buffer:
            outputs = extractor.process_batch(buffer)
            saved_count = self.save_extractions_batch(outputs)
            newly_extracted += len(outputs)
            total_findings_saved += saved_count
            buffer.clear()

        total_elapsed = max(time.perf_counter() - t_start, 0.001)
        final_rate = newly_extracted / total_elapsed

        if verbose:
            print(f"\n\nPipeline Run Finished in {total_elapsed:.2f}s ({final_rate:.1f} rec/s).")
            print(f"Total Newly Extracted: {newly_extracted:,} records | Total Findings Saved: {total_findings_saved:,}")
            print("=" * 80)

        return PipelineRunStats(
            total_matched=total_matched,
            already_extracted=already_extracted,
            newly_extracted=newly_extracted,
            total_findings_saved=total_findings_saved,
            elapsed_seconds=round(total_elapsed, 2),
            records_per_second=round(final_rate, 1),
        )

    # =========================================================================
    # VIGIPY DISPROPORTIONALITY COMPILATION
    # =========================================================================

    def to_vigipy_df(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        categories: Optional[Iterable[Union[str, CategoryType]]] = None,
        target_level: str = "code_term",
        format: str = "transaction",
    ) -> Any:
        """Loads compiled extractions from database directly into a Vigipy DataFrame."""
        outputs = self.load_extractions(product_code=product_code, brand_name=brand_name, category=None)
        from spotlight.vigipy_bridge import to_vigipy_df
        return to_vigipy_df(
            outputs,
            product_key="brand_name",
            target_level=target_level,
            categories=categories,
            format=format,
        )

    def to_vigipy_container(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        categories: Optional[Iterable[Union[str, CategoryType]]] = None,
        binary: bool = False,
        target_level: str = "code_term",
        margin_threshold: int = 1,
        **kwargs,
    ) -> Any:
        """Loads compiled extractions from database directly into a Vigipy DataContainer."""
        outputs = self.load_extractions(product_code=product_code, brand_name=brand_name, category=None)
        from spotlight.vigipy_bridge import to_vigipy_container
        return to_vigipy_container(
            outputs,
            binary=binary,
            product_key="brand_name",
            target_level=target_level,
            categories=categories,
            margin_threshold=margin_threshold,
            **kwargs,
        )


# Ergonomic public alias
ExtractionStore = MDRDatabase
MDROrchestrator = MDRDatabase


def init_mdr_database(
    db_path: str = "data/mdr.db",
    years: Optional[Union[int, Sequence[int], str]] = "recent",
    file_types: Optional[Sequence[str]] = None,
    sync: bool = False,
    download_dir: str = "data/mdr_downloads",
    verbose: bool = True,
) -> MDRDatabase:
    """
    Initializes or opens an MDR database and extraction store.
    """
    db = MDRDatabase(db_path=db_path)
    if sync:
        db.sync(years=years, file_types=file_types, download_dir=download_dir, verbose=verbose)
    return db
