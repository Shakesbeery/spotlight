"""
Unit tests for Spotlight -> Vigipy Bridge (disproportionality analysis integration).
"""

import pytest
import spotlight
from spotlight.schemas import (
    MAUDERecordInput,
    CategoryType,
    TemporalTiming,
    MAUDEExtractionOutput,
)
from spotlight.vigipy_bridge import (
    to_vigipy_df,
    to_vigipy_container,
    run_disproportionality_scan,
)


@pytest.fixture
def sample_device_records():
    """Returns a cohort of multi-device MAUDE reports with distinct failure modes."""
    return [
        MAUDERecordInput(
            mdr_report_key="R-101",
            brand_name="Stapler-Pro",
            product_code="GAG",
            event_type="Injury",
            narrative_text="Intraoperatively, the surgical stapler misfired and had a failure to cut. Tissue laceration and acute hemorrhage.",
        ),
        MAUDERecordInput(
            mdr_report_key="R-102",
            brand_name="Stapler-Pro",
            product_code="GAG",
            event_type="Injury",
            narrative_text="The stapler misfired on the mesenteric tissue. Bleeding occurred requiring emergency laparotomy.",
        ),
        MAUDERecordInput(
            mdr_report_key="R-103",
            brand_name="Stapler-Lite",
            product_code="GAG",
            event_type="Malfunction",
            narrative_text="During surgery, the stapler blade jammed during firing. No patient injury occurred.",
        ),
        MAUDERecordInput(
            mdr_report_key="R-104",
            brand_name="Catheter-Flow",
            product_code="LIT",
            event_type="Injury",
            narrative_text="The balloon burst during inflation causing arterial dissection. Covered stent deployed.",
        ),
        MAUDERecordInput(
            mdr_report_key="R-105",
            brand_name="Catheter-Flow",
            product_code="LIT",
            event_type="Injury",
            narrative_text="Balloon burst under pressure causing coronary dissection. Emergency stenting performed.",
        ),
        MAUDERecordInput(
            mdr_report_key="R-106",
            brand_name="Implant-Guard",
            product_code="JWH",
            event_type="Malfunction",
            narrative_text="Dark particulate matter observed on implant surface prior to use. Cracked blister package.",
        ),
    ]


@pytest.fixture
def extracted_outputs(sample_device_records):
    """Processes the sample records through Spotlight."""
    extractor = spotlight.get_default_extractor()
    return [extractor.process_record(r) for r in sample_device_records]


def test_to_vigipy_df_transaction_format(extracted_outputs, sample_device_records):
    """Verifies transaction DataFrame creation with standard metadata."""
    df = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        product_key="brand_name",
        target_level="code_term",
        format="transaction",
    )
    assert not df.empty
    assert "report_id" in df.columns
    assert "product" in df.columns
    assert "finding" in df.columns
    assert "category" in df.columns
    assert "count" in df.columns

    # Verify brand names are preserved
    products = set(df["product"].unique())
    assert "Stapler-Pro" in products
    assert "Catheter-Flow" in products


def test_to_vigipy_df_product_key_strategies(extracted_outputs, sample_device_records):
    """Verifies different product grouping strategies: brand, product code, and combined."""
    # 1. Product Code grouping (FDA 3-letter code)
    df_pcode = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        product_key="product_code",
    )
    pcodes = set(df_pcode["product"].unique())
    assert "GAG" in pcodes
    assert "LIT" in pcodes

    # 2. Combined Brand + Code grouping
    df_combo = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        product_key="brand_code",
    )
    combos = set(df_combo["product"].unique())
    assert any("Stapler-Pro (GAG)" in c for c in combos)
    assert any("Catheter-Flow (LIT)" in c for c in combos)

    # 3. Custom callable grouping
    df_custom = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        product_key=lambda out, rec: f"CUSTOM_{rec.product_code}",
    )
    assert "CUSTOM_GAG" in set(df_custom["product"].unique())


def test_to_vigipy_df_target_level_granularity(extracted_outputs, sample_device_records):
    """Verifies finding string representations across ontology target levels."""
    # 1. Code only (e.g. A0502, E0506)
    df_code = to_vigipy_df(extracted_outputs, records=sample_device_records, target_level="code")
    findings = set(df_code["finding"].unique())
    assert any(f.startswith("A") or f.startswith("E") or f.startswith("C") for f in findings)

    # 2. Term only
    df_term = to_vigipy_df(extracted_outputs, records=sample_device_records, target_level="term")
    terms = set(df_term["finding"].unique())
    assert any("Cutting" in t or "Hemorrhage" in t or "Dissection" in t for t in terms)

    # 3. Component + Failure mode
    df_comp = to_vigipy_df(extracted_outputs, records=sample_device_records, target_level="component_failure")
    comp_findings = set(df_comp["finding"].unique())
    # Should include component-linked strings like 'balloon: ...' or 'blade: ...'
    assert any("balloon" in cf.lower() or "blade" in cf.lower() or "jaw" in cf.lower() for cf in comp_findings)


def test_to_vigipy_df_category_and_timing_filtering(extracted_outputs, sample_device_records):
    """Verifies scoping to specific categories or clinical timing."""
    # Scoping only to operational problems
    df_op = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        categories=[CategoryType.OPERATIONAL_PROBLEM],
    )
    assert set(df_op["category"].unique()) == {"OPERATIONAL_PROBLEM"}

    # Scoping only to adverse events
    df_ae = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        categories=[CategoryType.ADVERSE_EVENT],
    )
    assert set(df_ae["category"].unique()) == {"ADVERSE_EVENT"}


def test_to_vigipy_df_aggregated_format(extracted_outputs, sample_device_records):
    """Verifies aggregated frequency table format."""
    df_agg = to_vigipy_df(
        extracted_outputs,
        records=sample_device_records,
        product_key="brand_name",
        format="aggregated",
    )
    assert list(df_agg.columns) == ["product", "finding", "count"]
    # Aggregated counts should be >= 1
    assert (df_agg["count"] >= 1).all()


def test_deduplication_per_report(extracted_outputs):
    """Verifies that duplicate findings in a single report are deduplicated."""
    df_dedup = to_vigipy_df(extracted_outputs, deduplicate_per_report=True)
    # Each (report_id, product, finding) tuple should appear at most once
    grouped = df_dedup.groupby(["report_id", "product", "finding"]).size()
    assert (grouped == 1).all()


def test_vigipy_container_conversion(extracted_outputs, sample_device_records):
    """Verifies conversion into vigipy DataContainer (both aggregated and binary)."""
    pytest.importorskip("vigipy")

    # Aggregated container
    cont = to_vigipy_container(
        extracted_outputs,
        records=sample_device_records,
        binary=False,
        product_key="brand_name",
        margin_threshold=1,
    )
    assert hasattr(cont, "contingency")
    assert hasattr(cont, "data")
    assert cont.contingency.shape[0] >= 3  # Multiple products

    # Binary container (for LASSO)
    bin_cont = to_vigipy_container(
        extracted_outputs,
        records=sample_device_records,
        binary=True,
        product_key="brand_name",
        sparse=True,
    )
    assert hasattr(bin_cont, "data")
    assert hasattr(bin_cont, "product_features")
    assert hasattr(bin_cont, "event_outcomes")


def test_run_disproportionality_scan_algorithms(extracted_outputs, sample_device_records):
    """Verifies executing PRR, ROR, BCPNN, and GPS in one high-level call."""
    pytest.importorskip("vigipy")

    scan_res = run_disproportionality_scan(
        extracted_outputs,
        records=sample_device_records,
        methods=["prr", "ror", "bcpnn", "gps"],
        min_events=1,
        product_key="brand_name",
    )

    assert "prr" in scan_res
    assert "ror" in scan_res
    assert "bcpnn" in scan_res
    assert "gps" in scan_res

    # Check PRR signals
    prr = scan_res["prr"]
    assert hasattr(prr, "signals")
    assert "PRR" in prr.signals.columns

    # Check ROR signals
    ror = scan_res["ror"]
    assert hasattr(ror, "signals")
    assert "ROR" in ror.signals.columns

    # Check BCPNN
    bcpnn = scan_res["bcpnn"]
    assert "quantile" in bcpnn.all_signals.columns

    # Check GPS
    gps = scan_res["gps"]
    assert "log2" in gps.all_signals.columns


def test_empty_outputs_handling():
    """Verifies graceful handling of empty inputs."""
    empty_df = to_vigipy_df([], format="transaction")
    assert empty_df.empty
    assert "report_id" in empty_df.columns

    empty_agg = to_vigipy_df([], format="aggregated")
    assert empty_agg.empty
    assert "product" in empty_agg.columns
