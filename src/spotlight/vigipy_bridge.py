"""
Vigipy Bridge for Spotlight: Seamless integration between Spotlight NLP extractions
and the Vigipy disproportionality analysis library (v3.0+).

Provides zero-friction conversion of MAUDEExtractionOutput batches into:
1. Transactional & Aggregated DataFrames formatted for vigipy
2. Typed vigipy.data.DataContainer objects for frequentist, Bayesian, and multivariable models
3. One-line disproportionality scans across device cohorts (PRR, ROR, BCPNN, GPS, LASSO)
"""

from typing import (
    Sequence,
    Optional,
    Iterable,
    Union,
    Callable,
    Dict,
    Any,
    List,
    Set,
    Tuple,
)
from collections import defaultdict

from spotlight.schemas import (
    MAUDEExtractionOutput,
    MAUDERecordInput,
    ExtractedFinding,
    CategoryType,
    TemporalTiming,
)

# Optional dependency imports handled gracefully
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

try:
    import vigipy
    _HAS_VIGIPY = True
except ImportError:
    vigipy = None
    _HAS_VIGIPY = False


def _extract_target_string(finding: ExtractedFinding, target_level: str) -> str:
    """Extracts target string for adverse event/finding column based on requested granularity."""
    norm = finding.normalized_terms[0] if finding.normalized_terms else None
    code = norm.code if norm else finding.verbatim_span
    term = norm.preferred_term if norm else finding.verbatim_span

    if target_level == "code_term":
        return f"{code}: {term}"
    elif target_level == "code":
        return code
    elif target_level == "term":
        return term
    elif target_level == "component_failure":
        if finding.affected_component:
            fm = finding.failure_mode or term
            return f"{finding.affected_component}: {fm}"
        return term
    elif target_level == "failure_mode":
        return finding.failure_mode or term
    elif target_level == "verbatim":
        return finding.verbatim_span
    else:
        # Default fallback
        return f"{code}: {term}"


def _get_product_identifier(
    output: MAUDEExtractionOutput,
    record: Optional[MAUDERecordInput],
    product_key: Union[str, Callable[[MAUDEExtractionOutput, Optional[MAUDERecordInput]], str]],
) -> str:
    """Resolves the product identifier string for an extraction record."""
    if callable(product_key):
        return str(product_key(output, record))

    brand = (record.brand_name if record and record.brand_name else None) or output.brand_name
    pcode = (record.product_code if record and record.product_code else None) or output.product_code

    if product_key == "brand_name":
        return brand or "UNKNOWN_BRAND"
    elif product_key == "product_code":
        return pcode or "UNKNOWN_CODE"
    elif product_key == "brand_code":
        if brand and pcode:
            return f"{brand} ({pcode})"
        return brand or pcode or "UNKNOWN_PRODUCT"
    elif product_key in ("report_id", "mdr_report_key"):
        return output.mdr_report_key
    else:
        return brand or "UNKNOWN_PRODUCT"


def to_vigipy_df(
    outputs: Sequence[MAUDEExtractionOutput],
    records: Optional[Sequence[MAUDERecordInput]] = None,
    product_key: Union[str, Callable[[MAUDEExtractionOutput, Optional[MAUDERecordInput]], str]] = "brand_name",
    target_level: str = "code_term",
    categories: Optional[Iterable[Union[str, CategoryType]]] = None,
    timing: Optional[Iterable[Union[str, TemporalTiming]]] = None,
    min_confidence: float = 0.0,
    deduplicate_per_report: bool = True,
    format: str = "transaction",
) -> "pd.DataFrame":
    """
    Converts a sequence of Spotlight MAUDE extractions into a pandas DataFrame ready for Vigipy.

    Args:
        outputs: Sequence of MAUDEExtractionOutput objects from Spotlight.
        records: Optional matching sequence of original MAUDERecordInput objects (for metadata).
        product_key: Product grouping strategy:
            - 'brand_name': Use device trade/brand name (e.g. 'Stapler-Alpha')
            - 'product_code': Use FDA 3-letter product code (e.g. 'GAG')
            - 'brand_code': Combine both (e.g. 'Stapler-Alpha (GAG)')
            - Custom callable: fn(output, record) -> str
        target_level: Granularity of adverse event / failure finding:
            - 'code_term': Standard IMDRF/MedDRA 'Code: Preferred Term' (default)
            - 'code': Pure ontology code (e.g. 'A0502')
            - 'term': Pure ontology term (e.g. 'Cutting Problem')
            - 'component_failure': Linked component + failure (e.g. 'jaw: jammed')
            - 'failure_mode': Granular physical failure mechanism
            - 'verbatim': Exact verbatim narrative span
        categories: Optional list/set of CategoryType or strings to filter (e.g. [CategoryType.OPERATIONAL_PROBLEM]).
            If None, all categories are included.
        timing: Optional filter for TemporalTiming (e.g. [TemporalTiming.INTRA_USE]).
        min_confidence: Minimum extraction confidence score threshold [0.0 - 1.0].
        deduplicate_per_report: If True (default), identical findings within the same MDR report key
            are counted only once per report to prevent narrative repetition bias.
        format: Output format:
            - 'transaction': One row per report-product-finding occurrence (with metadata)
            - 'aggregated': Aggregated product x finding frequency counts

    Returns:
        pandas.DataFrame formatted with standard column names for Vigipy.
    """
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for to_vigipy_df(). "
            "Please install it via 'pip install pandas' or 'pip install spotlight-nlp[vigipy]'."
        )

    # Convert category filter to set of enum values or string values
    cat_filter = None
    if categories is not None:
        cat_filter = {c.value if isinstance(c, CategoryType) else str(c) for c in categories}

    timing_filter = None
    if timing is not None:
        timing_filter = {t.value if isinstance(t, TemporalTiming) else str(t) for t in timing}

    record_map: Dict[str, MAUDERecordInput] = {}
    if records:
        for r in records:
            if r.mdr_report_key:
                record_map[r.mdr_report_key] = r

    rows: List[Dict[str, Any]] = []

    for idx, out in enumerate(outputs):
        rec = record_map.get(out.mdr_report_key)
        if rec is None and records and idx < len(records):
            rec = records[idx]

        prod_id = _get_product_identifier(out, rec, product_key)

        # Collect all findings across buckets
        category_buckets: List[Tuple[CategoryType, List[ExtractedFinding]]] = [
            (CategoryType.OPERATIONAL_PROBLEM, out.operational_problems),
            (CategoryType.MANUFACTURING_ISSUE, out.manufacturing_issues),
            (CategoryType.ADVERSE_EVENT, out.adverse_events),
            (CategoryType.CLINICAL_INTERVENTIONS, out.clinical_interventions),
            (CategoryType.OTHER, out.other),
        ]

        seen_in_report: Set[str] = set()

        for cat_enum, finding_list in category_buckets:
            cat_val = cat_enum.value
            if cat_filter and cat_val not in cat_filter:
                continue

            for f in finding_list:
                if f.confidence < min_confidence:
                    continue

                if timing_filter and f.temporal_timing.value not in timing_filter:
                    continue

                target_str = _extract_target_string(f, target_level)

                if deduplicate_per_report:
                    dedup_key = f"{out.mdr_report_key}||{prod_id}||{target_str}"
                    if dedup_key in seen_in_report:
                        continue
                    seen_in_report.add(dedup_key)

                norm = f.normalized_terms[0] if f.normalized_terms else None
                rows.append({
                    "report_id": out.mdr_report_key,
                    "product": prod_id,
                    "finding": target_str,
                    "category": cat_val,
                    "code": norm.code if norm else "",
                    "term": norm.preferred_term if norm else "",
                    "timing": f.temporal_timing.value,
                    "affected_component": f.affected_component or "",
                    "count": 1,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        # Return empty structured dataframe
        if format == "aggregated":
            return pd.DataFrame(columns=["product", "finding", "count"])
        return pd.DataFrame(columns=[
            "report_id", "product", "finding", "category",
            "code", "term", "timing", "affected_component", "count"
        ])

    if format == "aggregated":
        agg_df = (
            df.groupby(["product", "finding"], as_index=False)["count"]
            .sum()
            .reset_index(drop=True)
        )
        return agg_df

    return df


def to_vigipy_container(
    outputs: Sequence[MAUDEExtractionOutput],
    records: Optional[Sequence[MAUDERecordInput]] = None,
    binary: bool = False,
    product_key: Union[str, Callable[[MAUDEExtractionOutput, Optional[MAUDERecordInput]], str]] = "brand_name",
    target_level: str = "code_term",
    categories: Optional[Iterable[Union[str, CategoryType]]] = None,
    timing: Optional[Iterable[Union[str, TemporalTiming]]] = None,
    min_confidence: float = 0.0,
    deduplicate_per_report: bool = True,
    margin_threshold: int = 1,
    sparse: bool = True,
    covariate_labels: Optional[List[str]] = None,
    **convert_kwargs,
) -> Any:
    """
    Transforms Spotlight extractions into a ready-to-analyze Vigipy DataContainer.

    Args:
        outputs: Sequence of Spotlight MAUDE extraction outputs.
        records: Optional matching input records with brand/product metadata.
        binary: If True, calls vigipy.convert_binary() (for LASSO / longitudinal models).
            If False (default), calls vigipy.convert() for frequentist and Bayesian models.
        product_key: Strategy for product labels ('brand_name', 'product_code', 'brand_code', or callable).
        target_level: Finding target granularity ('code_term', 'code', 'term', 'component_failure').
        categories: Optional category filter (e.g. [CategoryType.OPERATIONAL_PROBLEM]).
        timing: Optional timing filter.
        min_confidence: Confidence threshold.
        deduplicate_per_report: Count each finding once per individual MDR report key.
        margin_threshold: Minimum row/column total count threshold for vigipy.convert().
        sparse: Whether to build sparse CSR contingency matrices in vigipy.convert_binary().
        covariate_labels: Optional list of covariate column names for binary conversion.
        **convert_kwargs: Additional keyword arguments forwarded to vigipy.convert/convert_binary.

    Returns:
        vigipy.data.DataContainer instance ready for PRR, ROR, BCPNN, GPS, or LASSO.
    """
    if not _HAS_VIGIPY:
        raise ImportError(
            "vigipy is required for to_vigipy_container(). "
            "Please install it via 'pip install vigipy' or install spotlight with 'pip install spotlight-nlp[vigipy]'."
        )

    if binary:
        df = to_vigipy_df(
            outputs=outputs,
            records=records,
            product_key=product_key,
            target_level=target_level,
            categories=categories,
            timing=timing,
            min_confidence=min_confidence,
            deduplicate_per_report=deduplicate_per_report,
            format="transaction",
        )
        container = vigipy.convert_binary(
            df,
            product_label="product",
            ae_label="finding",
            report_id_label="report_id",
            covariate_labels=covariate_labels,
            sparse=sparse,
            **convert_kwargs,
        )
        # Ensure sparse DataFrames fill unobserved entries with 0.0 to prevent NaN propagation in LASSO
        if sparse and hasattr(container, "event_outcomes") and container.event_outcomes is not None:
            try:
                container.event_outcomes = container.event_outcomes.fillna(0.0)
            except Exception:
                pass
        if sparse and hasattr(container, "product_features") and container.product_features is not None:
            try:
                container.product_features = container.product_features.fillna(0.0)
            except Exception:
                pass
        return container
    else:
        df = to_vigipy_df(
            outputs=outputs,
            records=records,
            product_key=product_key,
            target_level=target_level,
            categories=categories,
            timing=timing,
            min_confidence=min_confidence,
            deduplicate_per_report=deduplicate_per_report,
            format="aggregated",
        )
        return vigipy.convert(
            df,
            product_label="product",
            ae_label="finding",
            count_label="count",
            margin_threshold=margin_threshold,
            **convert_kwargs,
        )


def run_disproportionality_scan(
    outputs: Sequence[MAUDEExtractionOutput],
    records: Optional[Sequence[MAUDERecordInput]] = None,
    methods: Sequence[str] = ("prr", "ror", "bcpnn", "gps"),
    min_events: int = 1,
    product_key: str = "brand_name",
    target_level: str = "code_term",
    categories: Optional[Iterable[Union[str, CategoryType]]] = None,
    margin_threshold: int = 1,
    **method_kwargs,
) -> Dict[str, Any]:
    """
    Runs a full disproportionality surveillance scan on Spotlight extractions across a batch of devices.

    Args:
        outputs: Spotlight extraction outputs.
        records: Optional input records.
        methods: List of disproportionality algorithms to run:
            'prr', 'ror', 'rfet', 'bcpnn', 'gps', 'lasso'.
        min_events: Minimum count threshold for signal detection.
        product_key: Product identifier grouping ('brand_name', 'product_code', 'brand_code').
        target_level: Finding target granularity ('code_term', 'code', 'term', 'component_failure').
        categories: Optional category scoping (e.g. [CategoryType.OPERATIONAL_PROBLEM]).
        margin_threshold: Minimum margin count threshold.
        **method_kwargs: Additional parameters passed to vigipy analysis functions.

    Returns:
        Dictionary mapping method name to its corresponding vigipy.AnalysisResult, plus 'container'.
    """
    if not _HAS_VIGIPY:
        raise ImportError(
            "vigipy is required for run_disproportionality_scan(). "
            "Please install it via 'pip install vigipy'."
        )

    results: Dict[str, Any] = {}
    normalized_methods = [m.lower().strip() for m in methods]

    # Needs aggregated container if frequentist/Bayesian methods are requested
    needs_aggregated = any(m in ("prr", "ror", "rfet", "bcpnn", "gps") for m in normalized_methods)
    needs_binary = "lasso" in normalized_methods

    container = None
    if needs_aggregated:
        container = to_vigipy_container(
            outputs=outputs,
            records=records,
            binary=False,
            product_key=product_key,
            target_level=target_level,
            categories=categories,
            margin_threshold=margin_threshold,
        )
        results["container"] = container

        if "prr" in normalized_methods:
            results["prr"] = vigipy.prr(
                container,
                min_events=min_events,
                decision_metric=method_kwargs.get("decision_metric", "rank"),
                decision_thres=method_kwargs.get("prr_threshold", 1.0),
            )

        if "ror" in normalized_methods:
            results["ror"] = vigipy.ror(
                container,
                min_events=min_events,
                decision_metric=method_kwargs.get("decision_metric", "rank"),
                decision_thres=method_kwargs.get("ror_threshold", 1.0),
            )

        if "rfet" in normalized_methods:
            results["rfet"] = vigipy.rfet(
                container,
                min_events=min_events,
            )

        if "bcpnn" in normalized_methods:
            results["bcpnn"] = vigipy.bcpnn(
                container,
                min_events=min_events,
                ranking_statistic=method_kwargs.get("bcpnn_statistic", "quantile"),
            )

        if "gps" in normalized_methods:
            results["gps"] = vigipy.gps(
                container,
                min_events=min_events,
            )

    if needs_binary:
        bin_container = to_vigipy_container(
            outputs=outputs,
            records=records,
            binary=True,
            product_key=product_key,
            target_level=target_level,
            categories=categories,
            sparse=method_kwargs.get("sparse", True),
        )
        results["binary_container"] = bin_container
        results["lasso"] = vigipy.lasso(
            bin_container,
            min_events=min_events,
            relaxed=method_kwargs.get("lasso_relaxed", True),
            decision_metric=method_kwargs.get("lasso_decision_metric", "lower_bound"),
            lasso_thresh=method_kwargs.get("lasso_threshold", 0.0),
        )

    return results
