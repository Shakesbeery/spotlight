"""
Project Management Module for Spotlight.

Enables creating and managing isolated project workspaces with:
1. Dedicated project directories with manifest (project.json).
2. Project-isolated extraction databases (extractions.db).
3. Cross-project contamination guards: Prevents accidentally saving or splitting
   findings across projects during interactive / live coding sessions.
4. Project-scoped delta pipelines: Anti-joins against the project's own extraction
   store while reading from a shared, read-only raw MDR database.
5. Project-scoped Vigipy exports and analytical artifacts.
"""

import os
import json
import time
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Generator,
    Union,
    Set,
    Callable,
    Iterable,
)
from dataclasses import dataclass
from spotlight.schemas import (
    MAUDERecordInput,
    MAUDEExtractionOutput,
    CategoryType,
)
from spotlight.mdr_database import MDRDatabase, PipelineRunStats


class ProjectError(Exception):
    """Base exception for Spotlight project workspace errors."""
    pass


class ProjectMismatchError(ProjectError):
    """Raised when an operation attempts to mix or contaminate data from different projects."""
    pass


class ProjectNotFoundError(ProjectError):
    """Raised when a requested project workspace does not exist."""
    pass


class ProjectActiveConflictError(ProjectError):
    """Raised when an illegal project context switch is attempted in an active session."""
    pass


class SpotlightProject:
    """
    Manages an isolated project workspace for MDR queries, NLP extractions, and surveillance.
    
    Each project encapsulates:
    - Dedicated filesystem directory: <base_dir>/<name>/
    - Project configuration manifest: project.json
    - Dedicated SQLite extraction database: extractions.db
    - Dedicated exports directory: exports/
    - Shared reference to central raw MDR database (read-only)
    """

    _thread_local = threading.local()

    def __init__(
        self,
        name: str,
        base_dir: str = "projects",
        raw_db: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exist_ok: bool = True,
    ):
        clean_name = self._validate_project_name(name)
        self.name = clean_name
        self.base_dir = os.path.abspath(base_dir)
        self.project_dir = os.path.join(self.base_dir, self.name)
        self.manifest_path = os.path.join(self.project_dir, "project.json")
        self.extractions_db_path = os.path.join(self.project_dir, "extractions.db")
        self.exports_dir = os.path.join(self.project_dir, "exports")

        self._raw_db_instance: Optional[MDRDatabase] = None
        self._extractions_db_instance: Optional[MDRDatabase] = None

        if not os.path.exists(self.project_dir):
            self.raw_db_path = os.path.abspath(raw_db or "data/maude.db")
            self._create_workspace(description=description, tags=tags)
        else:
            if not exist_ok:
                raise ProjectError(f"Project '{self.name}' already exists at {self.project_dir}")
            self._load_workspace(raw_db=raw_db, description=description, tags=tags)

    @staticmethod
    def _validate_project_name(name: str) -> str:
        clean = name.strip()
        if not clean:
            raise ValueError("Project name cannot be empty.")
        # Allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r"^[A-Za-z0-9_-]+$", clean):
            raise ValueError(f"Invalid project name '{name}'. Use letters, numbers, hyphens, and underscores only.")
        return clean

    def _create_workspace(self, description: Optional[str] = None, tags: Optional[List[str]] = None):
        """Initializes directory structure, manifest, and isolated extraction store."""
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        manifest_data = {
            "name": self.name,
            "created_at": now_iso,
            "updated_at": now_iso,
            "description": description or f"Spotlight study workspace for {self.name}",
            "tags": tags or [],
            "raw_db_path": self.raw_db_path,
            "extractions_db_path": self.extractions_db_path,
            "pipeline_runs": [],
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Initialize project extraction database schema
        db = MDRDatabase(db_path=self.extractions_db_path)
        db.close()

    def _load_workspace(self, raw_db: Optional[str] = None, description: Optional[str] = None, tags: Optional[List[str]] = None):
        """Loads existing workspace and refreshes manifest if updated."""
        if not os.path.exists(self.manifest_path):
            self.raw_db_path = os.path.abspath(raw_db or "data/maude.db")
            self._create_workspace(description=description, tags=tags)
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        if raw_db is not None:
            self.raw_db_path = os.path.abspath(raw_db)
            manifest_data["raw_db_path"] = self.raw_db_path
        else:
            self.raw_db_path = manifest_data.get("raw_db_path", os.path.abspath("data/maude.db"))

        updated = False
        if description and manifest_data.get("description") != description:
            manifest_data["description"] = description
            updated = True
        if tags is not None and manifest_data.get("tags") != tags:
            manifest_data["tags"] = tags
            updated = True

        if updated:
            manifest_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

    @property
    def manifest(self) -> Dict[str, Any]:
        """Returns the current project manifest."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"name": self.name}

    def _record_pipeline_run(self, stats: PipelineRunStats, filters: Dict[str, Any]):
        """Records a completed pipeline execution into the project manifest."""
        try:
            m = self.manifest
            runs = m.get("pipeline_runs", [])
            runs.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filters": filters,
                "total_matched": stats.total_matched,
                "already_extracted": stats.already_extracted,
                "newly_extracted": stats.newly_extracted,
                "total_findings_saved": stats.total_findings_saved,
                "elapsed_seconds": stats.elapsed_seconds,
                "records_per_second": stats.records_per_second,
            })
            m["pipeline_runs"] = runs[-50:]  # Keep last 50 runs
            m["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        except Exception:
            pass

    # =========================================================================
    # LIVE SESSION & CONTEXT MANAGEMENT
    # =========================================================================

    @classmethod
    def get_active(cls) -> Optional["SpotlightProject"]:
        """Returns the currently active SpotlightProject for this execution context."""
        return getattr(cls._thread_local, "active_project", None)

    @classmethod
    def set_active(cls, project: Optional["SpotlightProject"]):
        """Explicitly sets or clears the active project in the current session."""
        cls._thread_local.active_project = project

    def __enter__(self) -> "SpotlightProject":
        current = SpotlightProject.get_active()
        if current is not None and current.name != self.name:
            raise ProjectActiveConflictError(
                f"Cannot activate project '{self.name}': Project '{current.name}' is already active in this session. "
                f"Exit the current project context or call project.close() before switching."
            )
        self._prev_active = current
        SpotlightProject.set_active(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        prev = getattr(self, "_prev_active", None)
        SpotlightProject.set_active(prev)
        self.close()

    # =========================================================================
    # DATABASE CONNECTIONS
    # =========================================================================

    def get_raw_db(self) -> MDRDatabase:
        """Returns connection to shared raw MDR database (master, device, text)."""
        if self._raw_db_instance is None:
            self._raw_db_instance = MDRDatabase(db_path=self.raw_db_path)
        return self._raw_db_instance

    def get_extraction_store(self) -> MDRDatabase:
        """Returns connection to the project's private extraction database."""
        if self._extractions_db_instance is None:
            self._extractions_db_instance = MDRDatabase(db_path=self.extractions_db_path)
        return self._extractions_db_instance

    def close(self):
        """Releases database connections and forces WAL checkpoint."""
        if self._raw_db_instance is not None:
            try:
                self._raw_db_instance.close()
            except Exception:
                pass
            self._raw_db_instance = None

        if self._extractions_db_instance is not None:
            try:
                self._extractions_db_instance.close()
            except Exception:
                pass
            self._extractions_db_instance = None

    # =========================================================================
    # PROJECT-SCOPED EXTRACTION PERSISTENCE & CROSS-CONTAMINATION GUARD
    # =========================================================================

    def _tag_output(self, output: MAUDEExtractionOutput) -> MAUDEExtractionOutput:
        """Tags extraction output with this project's name."""
        if output.project_name == self.name:
            return output
        # Return shallow copy with project_name set
        data = output.model_dump()
        data["project_name"] = self.name
        return MAUDEExtractionOutput(**data)

    def save_extractions(
        self,
        outputs: Sequence[MAUDEExtractionOutput],
        allow_cross_project: bool = False,
    ) -> int:
        """
        Durably persists extraction outputs into this project's isolated extraction store.
        
        GUARD CHECK:
        If any extraction output is tagged with a DIFFERENT project name, raises
        ProjectMismatchError to prevent accidental cross-project splitting or contamination.
        Set allow_cross_project=True to explicitly permit cross-project imports.
        """
        if not outputs:
            return 0

        # Cross-project contamination check
        if not allow_cross_project:
            mismatches = [
                (out.mdr_report_key, out.project_name)
                for out in outputs
                if out.project_name is not None and out.project_name != self.name
            ]
            if mismatches:
                bad_key, bad_proj = mismatches[0]
                raise ProjectMismatchError(
                    f"Cross-project contamination prevented! "
                    f"Report '{bad_key}' belongs to project '{bad_proj}', but active project is '{self.name}'. "
                    f"Accidentally saving extractions into the wrong project workspace is prohibited. "
                    f"To explicitly import cross-project results, pass allow_cross_project=True."
                )

        # Tag untagged outputs with this project name
        tagged = [self._tag_output(out) for out in outputs]
        store = self.get_extraction_store()
        return store.save_extractions_batch(tagged)

    def load_extractions(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        category: Optional[Union[str, CategoryType]] = None,
        code: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MAUDEExtractionOutput]:
        """Loads extractions stored strictly within this project workspace."""
        store = self.get_extraction_store()
        return store.load_extractions(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            category=category,
            code=code,
            limit=limit,
        )

    def stream_extractions(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        event_type: Optional[str] = None,
        category: Optional[Union[str, CategoryType]] = None,
        code: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> Generator[MAUDEExtractionOutput, None, None]:
        """Streams extractions stored strictly within this project workspace."""
        store = self.get_extraction_store()
        return store.stream_extractions(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            category=category,
            code=code,
            limit=limit,
            chunk_size=chunk_size,
        )

    def get_extracted_keys(self) -> Set[str]:
        """Returns all report keys extracted and saved within this project."""
        store = self.get_extraction_store()
        return store.get_extracted_keys()

    def count_extracted(self) -> int:
        """Returns count of reports extracted in this project."""
        store = self.get_extraction_store()
        stats = store.get_extraction_stats()
        return stats.get("total_reports_extracted", 0)

    # =========================================================================
    # PROJECT-SCOPED DELTA ORCHESTRATION PIPELINE
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
        allow_cross_project: bool = False,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineRunStats:
        """
        Orchestrates an end-to-end Query -> Extract -> Save pipeline strictly scoped to this project.
        
        1. Queries candidate records from the shared raw MDR database.
        2. If skip_already_extracted=True, anti-joins against THIS project's extractions.db
           so only delta records unextracted in THIS project are processed.
        3. Streams in constant memory (<100MB RAM), extracts findings with Spotlight,
           and tags each result with self.name.
        4. Saves findings into this project's extractions.db.
        """
        if extractor is None:
            import spotlight
            extractor = spotlight.get_default_extractor()

        raw_db = self.get_raw_db()
        project_store = self.get_extraction_store()

        # Step 1: Pre-flight count of matched vs extracted in this project
        total_matched = raw_db.count_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            only_unextracted=False,
        )

        extracted_keys = self.get_extracted_keys() if skip_already_extracted else set()
        already_extracted = len(extracted_keys) if skip_already_extracted else 0
        unextracted_total = max(total_matched - already_extracted, 0)

        if verbose:
            print("=" * 80)
            print(f"SPOTLIGHT PROJECT PIPELINE: '{self.name}'")
            print(f"Project Workspace: {self.project_dir}")
            print(f"Filters: Product Code: {product_code or 'ALL'} | Brand: {brand_name or 'ALL'} | Event Type: {event_type or 'ALL'}")
            print(f"Inventory: {total_matched:,} total records matching query in raw DB.")
            print(f"           {already_extracted:,} records already extracted in this project (skipped).")
            print(f"           {unextracted_total:,} delta records queued for extraction.")
            print("=" * 80)

        if skip_already_extracted and unextracted_total == 0:
            if verbose:
                print("All matching records have already been extracted in this project! Zero redundant compute.")
            stats = PipelineRunStats(
                total_matched=total_matched,
                already_extracted=already_extracted,
                newly_extracted=0,
                total_findings_saved=0,
                elapsed_seconds=0.0,
                records_per_second=0.0,
            )
            self._record_pipeline_run(stats, {
                "product_code": product_code,
                "brand_name": brand_name,
                "event_type": event_type,
                "query": query,
            })
            return stats

        t_start = time.perf_counter()
        newly_extracted = 0
        total_findings_saved = 0

        # Stream records from raw database
        stream = raw_db.stream_records(
            product_code=product_code,
            brand_name=brand_name,
            event_type=event_type,
            query=query,
            only_unextracted=False,
            limit=None,
            chunk_size=chunk_size,
        )

        buffer: List[MAUDERecordInput] = []

        for record in stream:
            # Project-specific delta filter
            if skip_already_extracted and record.mdr_report_key in extracted_keys:
                continue

            buffer.append(record)

            if len(buffer) >= chunk_size:
                outputs = extractor.process_batch(buffer)
                # Tag outputs with this project name
                tagged = [self._tag_output(o) for o in outputs]
                saved_count = project_store.save_extractions_batch(tagged)

                # Add newly processed keys to local set
                for o in tagged:
                    extracted_keys.add(o.mdr_report_key)

                newly_extracted += len(tagged)
                total_findings_saved += saved_count

                if verbose:
                    elapsed = max(time.perf_counter() - t_start, 0.001)
                    rate = newly_extracted / elapsed
                    target_num = min(unextracted_total, limit) if limit else unextracted_total
                    print(
                        f"\r[{self.name}] Extracted: {newly_extracted:,}/{target_num:,} | "
                        f"Findings: {total_findings_saved:,} | "
                        f"Speed: {rate:.1f} rec/s",
                        end="", flush=True
                    )

                if progress_callback:
                    progress_callback(newly_extracted, unextracted_total)

                buffer.clear()
                if limit is not None and newly_extracted >= limit:
                    break

        # Flush remaining buffer
        if buffer and (limit is None or newly_extracted < limit):
            rem_limit = limit - newly_extracted if limit else len(buffer)
            outputs = extractor.process_batch(buffer[:rem_limit])
            tagged = [self._tag_output(o) for o in outputs]
            saved_count = project_store.save_extractions_batch(tagged)
            newly_extracted += len(tagged)
            total_findings_saved += saved_count
            buffer.clear()

        total_elapsed = max(time.perf_counter() - t_start, 0.001)
        final_rate = newly_extracted / total_elapsed

        if verbose:
            print(f"\n\nProject '{self.name}' Pipeline Finished in {total_elapsed:.2f}s ({final_rate:.1f} rec/s).")
            print(f"Newly Extracted: {newly_extracted:,} | Findings Saved: {total_findings_saved:,}")
            print(f"Extractions Database: {self.extractions_db_path}")
            print("=" * 80)

        stats = PipelineRunStats(
            total_matched=total_matched,
            already_extracted=already_extracted,
            newly_extracted=newly_extracted,
            total_findings_saved=total_findings_saved,
            elapsed_seconds=round(total_elapsed, 2),
            records_per_second=round(final_rate, 1),
        )

        self._record_pipeline_run(stats, {
            "product_code": product_code,
            "brand_name": brand_name,
            "event_type": event_type,
            "query": query,
        })

        return stats

    # =========================================================================
    # PROJECT-SCOPED VIGIPY DISPROPORTIONALITY & EXPORTS
    # =========================================================================

    def to_vigipy_df(
        self,
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        categories: Optional[Iterable[Union[str, CategoryType]]] = None,
        target_level: str = "code_term",
        format: str = "transaction",
    ) -> Any:
        """Compiles extractions stored in THIS project directly into a Vigipy DataFrame."""
        outputs = self.load_extractions(product_code=product_code, brand_name=brand_name)
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
        """Compiles extractions stored in THIS project directly into a Vigipy DataContainer."""
        outputs = self.load_extractions(product_code=product_code, brand_name=brand_name)
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

    def export_vigipy_dataset(
        self,
        filename: str = "vigipy_dataset.parquet",
        product_code: Optional[str] = None,
        brand_name: Optional[str] = None,
        categories: Optional[Iterable[Union[str, CategoryType]]] = None,
        target_level: str = "code_term",
        format: str = "transaction",
    ) -> str:
        """
        Exports compiled project findings into a parquet or CSV file inside exports/.
        Returns the absolute path to the exported dataset.
        """
        df = self.to_vigipy_df(
            product_code=product_code,
            brand_name=brand_name,
            categories=categories,
            target_level=target_level,
            format=format,
        )
        dest_path = os.path.join(self.exports_dir, filename)
        if filename.endswith(".parquet"):
            try:
                df.to_parquet(dest_path, index=False)
            except ImportError:
                csv_path = os.path.splitext(dest_path)[0] + ".csv"
                import warnings
                warnings.warn(f"Parquet engine (pyarrow/fastparquet) unavailable; falling back to CSV export at {csv_path}")
                df.to_csv(csv_path, index=False)
                return csv_path
        elif filename.endswith(".csv"):
            df.to_csv(dest_path, index=False)
        else:
            try:
                df.to_parquet(dest_path + ".parquet", index=False)
                dest_path += ".parquet"
            except ImportError:
                dest_path += ".csv"
                df.to_csv(dest_path, index=False)
        return dest_path

    # =========================================================================
    # RAW METADATA LOOKUP FROM PROJECT
    # =========================================================================

    def get_raw_metadata(self, mdr_report_key: str) -> Optional[Dict[str, Any]]:
        """Looks up raw MDR master/device/narrative metadata from the shared raw database."""
        raw_db = self.get_raw_db()
        return raw_db.get_metadata(mdr_report_key)

    def stats(self) -> Dict[str, Any]:
        """Returns comprehensive project statistics."""
        store = self.get_extraction_store()
        ext_stats = store.get_extraction_stats()
        raw_db = self.get_raw_db()

        exports = os.listdir(self.exports_dir) if os.path.exists(self.exports_dir) else []
        db_size_bytes = os.path.getsize(self.extractions_db_path) if os.path.exists(self.extractions_db_path) else 0

        return {
            "name": self.name,
            "project_dir": self.project_dir,
            "description": self.manifest.get("description", ""),
            "created_at": self.manifest.get("created_at", ""),
            "raw_db_path": self.raw_db_path,
            "extractions_db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "total_extracted_reports": ext_stats.get("total_reports_extracted", 0),
            "total_findings_saved": ext_stats.get("total_findings_saved", 0),
            "findings_by_category": ext_stats.get("findings_by_category", {}),
            "top_imdrf_codes": ext_stats.get("top_imdrf_codes", []),
            "total_exports": len(exports),
            "exports": exports,
        }

    # =========================================================================
    # STATIC / CLASS WORKSPACE MANAGEMENT
    # =========================================================================

    @classmethod
    def list_projects(cls, base_dir: str = "projects") -> List[Dict[str, Any]]:
        """Lists all existing project workspaces in base_dir."""
        base_path = os.path.abspath(base_dir)
        if not os.path.exists(base_path):
            return []

        projects = []
        for entry in os.listdir(base_path):
            p_dir = os.path.join(base_path, entry)
            manifest = os.path.join(p_dir, "project.json")
            if os.path.isdir(p_dir) and os.path.exists(manifest):
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    db_file = os.path.join(p_dir, "extractions.db")
                    data["size_mb"] = round(os.path.getsize(db_file) / (1024 * 1024), 2) if os.path.exists(db_file) else 0
                    projects.append(data)
                except Exception:
                    pass

        return sorted(projects, key=lambda x: x.get("name", ""))

    @classmethod
    def open(
        cls,
        name: str,
        base_dir: str = "projects",
        raw_db: Optional[str] = None,
    ) -> "SpotlightProject":
        """Opens an existing project workspace, raising ProjectNotFoundError if it does not exist."""
        clean_name = cls._validate_project_name(name)
        p_dir = os.path.join(os.path.abspath(base_dir), clean_name)
        if not os.path.exists(p_dir):
            raise ProjectNotFoundError(f"Project '{name}' does not exist at {p_dir}. Use SpotlightProject(...) to create it.")
        return cls(name=clean_name, base_dir=base_dir, raw_db=raw_db, exist_ok=True)
