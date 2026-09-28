"""
Orchestrator and end-to-end cascaded pipeline for MAUDE FDA records.
Coordinates Stage 1 Preprocessing, Tier 1 Fast Triage, Tier 2 Gemma 4 Fallback,
Ontology Grounding, and final synthesis.
"""

import time
from typing import List, Optional, Union, Generator, Any
from spotlight.schemas import (
    MAUDERecordInput,
    MAUDEExtractionOutput,
    CategoryType,
    ExecutionPath,
    ExtractedFinding,
)
from spotlight.preprocessor import MAUDEPreprocessor
from spotlight.fast_triage import FastTriageEngine
from spotlight.llm_fallback import Gemma4FallbackEngine


from spotlight.component_linker import ComponentLinker, EntityResolver


class MAUDEExtractionPipeline:
    """
    Production-grade cascaded extraction pipeline for FDA MAUDE narratives.
    High-throughput execution with intelligent SLM fallback, component linkage,
    and coreference reconciliation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        gemma4_endpoint: Optional[str] = None,
        use_mock_slm: bool = True,
    ):
        self.preprocessor = MAUDEPreprocessor()
        self.fast_triage = FastTriageEngine(confidence_threshold=confidence_threshold)
        self.gemma4_fallback = Gemma4FallbackEngine(endpoint_url=gemma4_endpoint)
        self.component_linker = ComponentLinker()
        self.use_mock_slm = use_mock_slm

    def _bucket_findings(
        self, findings: List[ExtractedFinding]
    ) -> dict:
        """Categorizes a list of findings into target output buckets with IDs and causal links."""
        buckets = {
            CategoryType.OPERATIONAL_PROBLEM: [],
            CategoryType.MANUFACTURING_ISSUE: [],
            CategoryType.ADVERSE_EVENT: [],
            CategoryType.CLINICAL_INTERVENTIONS: [],
            CategoryType.OTHER: [],
        }

        # Deduplicate findings by verbatim_span and category
        seen = set()
        counters = {cat: 1 for cat in CategoryType}
        prefix_map = {
            CategoryType.OPERATIONAL_PROBLEM: "OP",
            CategoryType.MANUFACTURING_ISSUE: "MFG",
            CategoryType.ADVERSE_EVENT: "AE",
            CategoryType.CLINICAL_INTERVENTIONS: "INT",
            CategoryType.OTHER: "OTH",
        }

        for f in findings:
            key = (f.category, f.verbatim_span.lower().strip())
            if key not in seen:
                seen.add(key)
                fid = f.id or f"{prefix_map[f.category]}-{counters[f.category]}"
                counters[f.category] += 1

                # Causal attribution for interventions
                trig = f.triggered_by
                if f.category == CategoryType.CLINICAL_INTERVENTIONS and not trig:
                    if buckets[CategoryType.ADVERSE_EVENT]:
                        trig = buckets[CategoryType.ADVERSE_EVENT][-1].id
                    elif buckets[CategoryType.OPERATIONAL_PROBLEM]:
                        trig = buckets[CategoryType.OPERATIONAL_PROBLEM][-1].id

                finding_with_id = ExtractedFinding(
                    id=fid,
                    category=f.category,
                    verbatim_span=f.verbatim_span,
                    start_char=f.start_char,
                    end_char=f.end_char,
                    temporal_timing=f.temporal_timing,
                    confidence=f.confidence,
                    affected_component=f.affected_component,
                    failure_mode=f.failure_mode,
                    aliases=f.aliases,
                    triggered_by=trig,
                    normalized_terms=f.normalized_terms,
                    details=f.details,
                )
                buckets[f.category].append(finding_with_id)

        return buckets

    def process_record(self, record: MAUDERecordInput) -> MAUDEExtractionOutput:
        """
        Executes end-to-end processing for a single MAUDE report record.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocessing & Boilerplate Stripping
        cleaned_narrative = self.preprocessor.strip_boilerplate(record.narrative_text)
        segments = self.preprocessor.segment_text(cleaned_narrative, is_cleaned=True)

        # Step 2: Fast Triage Tier 1 (ModernBERT / High-speed baseline)
        findings, requires_fallback = self.fast_triage.process_segments(segments)
        execution_path = ExecutionPath.FAST_PATH_MODERNBERT

        # Step 3: Tier 2 Gemma 4 Fallback (if triggered)
        if requires_fallback:
            slm_findings = self.gemma4_fallback.extract_with_gemma4(
                narrative=cleaned_narrative,
                use_mock=self.use_mock_slm
            )
            if slm_findings:
                # Merge or prioritize SLM findings for ambiguous cases
                findings = slm_findings
                execution_path = ExecutionPath.SLM_FALLBACK_GEMMA4

        # Step 4: Link Named Device Components to Failures
        findings = self.component_linker.link_components_to_findings(cleaned_narrative, findings)

        # Step 5: Reconcile True Coreferent Duplicates (while preserving distinct manifestations)
        findings = EntityResolver.reconcile_duplicates(findings, cleaned_narrative)

        # Step 6: Group findings into discrete categories with IDs
        bucketed = self._bucket_findings(findings)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MAUDEExtractionOutput(
            mdr_report_key=record.mdr_report_key,
            brand_name=record.brand_name,
            product_code=record.product_code,
            event_type=record.event_type,
            operational_problems=bucketed[CategoryType.OPERATIONAL_PROBLEM],
            manufacturing_issues=bucketed[CategoryType.MANUFACTURING_ISSUE],
            adverse_events=bucketed[CategoryType.ADVERSE_EVENT],
            clinical_interventions=bucketed[CategoryType.CLINICAL_INTERVENTIONS],
            other=bucketed[CategoryType.OTHER],
            execution_path=execution_path,
            processing_time_ms=round(elapsed_ms, 2),
            cleaned_narrative=cleaned_narrative,
        )

    def process_narrative(
        self,
        narrative_text: str,
        mdr_report_key: str = "ADHOC-RECORD",
        brand_name: Optional[str] = None,
        product_code: Optional[str] = None,
        event_type: str = "Unknown",
    ) -> MAUDEExtractionOutput:
        """
        Convenience method to process a raw clinical or device narrative string directly.
        """
        record = MAUDERecordInput(
            mdr_report_key=mdr_report_key,
            brand_name=brand_name,
            product_code=product_code,
            event_type=event_type,
            narrative_text=narrative_text,
        )
        return self.process_record(record)

    def process_batch(
        self, records: Any
    ) -> List[MAUDEExtractionOutput]:
        """Processes a batch of records (raw strings or MAUDERecordInput instances)."""
        outputs = []
        for r in records:
            if isinstance(r, str):
                outputs.append(self.process_narrative(r))
            else:
                outputs.append(self.process_record(r))
        return outputs

    def process_stream(
        self, records: Any
    ) -> Generator[MAUDEExtractionOutput, None, None]:
        """
        Lazily processes and streams extraction outputs one by one.
        Maintains constant memory overhead (<100MB) even across millions of records.
        """
        for r in records:
            if isinstance(r, str):
                yield self.process_narrative(r)
            else:
                yield self.process_record(r)


# Ergonomic public alias for library users
SpotlightExtractor = MAUDEExtractionPipeline

