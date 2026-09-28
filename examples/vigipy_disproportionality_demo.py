"""
End-to-end Demonstration: Spotlight NLP Extraction to Vigipy Disproportionality Surveillance.

This script demonstrates:
1. Ingesting multi-device FDA MAUDE medical device narratives.
2. Extracting structured clinical and device findings grounded in IMDRF and MedDRA.
3. Converting extractions into Vigipy-ready DataFrames and typed DataContainers.
4. Executing frequentist (PRR, ROR), Bayesian (BCPNN, GPS/EBGM), and multivariable (LASSO)
   disproportionality algorithms to detect device safety signals.
"""

import spotlight
from spotlight.schemas import MAUDERecordInput, CategoryType
from spotlight.vigipy_bridge import (
    to_vigipy_df,
    to_vigipy_container,
    run_disproportionality_scan,
)

# Sample cross-device cohort
MAUDE_RECORDS = [
    # Surgical Staplers (Brand Alpha) - Misfire and cutting problems
    MAUDERecordInput(
        mdr_report_key="MDR-1001",
        brand_name="Stapler-Alpha",
        product_code="GAG",
        event_type="Injury",
        narrative_text="Intraoperatively, the surgical stapler misfired and had a complete failure to cut. Caused tissue laceration and acute hemorrhage.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1002",
        brand_name="Stapler-Alpha",
        product_code="GAG",
        event_type="Injury",
        narrative_text="The stapler misfired across the mesenteric tissue. Patient experienced severe bleeding requiring laparotomy.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1003",
        brand_name="Stapler-Alpha",
        product_code="GAG",
        event_type="Injury",
        narrative_text="Surgical stapler demonstrated failure to cut and jammed jaw. Acute hemorrhage occurred.",
    ),
    # Surgical Staplers (Brand Beta) - Jams without misfire
    MAUDERecordInput(
        mdr_report_key="MDR-1004",
        brand_name="Stapler-Beta",
        product_code="GAG",
        event_type="Malfunction",
        narrative_text="The stapler jammed during firing. No patient injury occurred. Swapped device.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1005",
        brand_name="Stapler-Beta",
        product_code="GAG",
        event_type="Malfunction",
        narrative_text="During surgery, mechanical jam occurred with the stapler. Anvil would not release.",
    ),
    # Balloon Catheters (Brand Gamma) - Balloon burst and dissection
    MAUDERecordInput(
        mdr_report_key="MDR-1006",
        brand_name="Catheter-Gamma",
        product_code="LIT",
        event_type="Injury",
        narrative_text="The balloon burst during inflation causing arterial dissection. Covered stent deployed.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1007",
        brand_name="Catheter-Gamma",
        product_code="LIT",
        event_type="Injury",
        narrative_text="Balloon burst under pressure causing coronary dissection. Emergency stenting performed.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1008",
        brand_name="Catheter-Gamma",
        product_code="LIT",
        event_type="Injury",
        narrative_text="Catheter balloon ruptured in vivo causing vessel perforation and severe hypotension.",
    ),
    # Orthopedic Implants (Brand Delta) - Packaging & Particulate
    MAUDERecordInput(
        mdr_report_key="MDR-1009",
        brand_name="Implant-Delta",
        product_code="JWH",
        event_type="Malfunction",
        narrative_text="Prior to use, dark particulate matter was observed on the implant surface. Sterile blister tray was cracked.",
    ),
    MAUDERecordInput(
        mdr_report_key="MDR-1010",
        brand_name="Implant-Delta",
        product_code="JWH",
        event_type="Malfunction",
        narrative_text="Visible dark particulate matter inside packaging prior to surgery. Compromised seal on blister package.",
    ),
]


def main():
    print("=" * 80)
    print("SPOTLIGHT -> VIGIPY DISPROPORTIONALITY ANALYSIS DEMONSTRATION")
    print("=" * 80)

    # 1. High-throughput extraction across the device cohort
    print("\n1. Running Spotlight extraction on 10 multi-device MAUDE reports...")
    outputs = spotlight.extract_batch(MAUDE_RECORDS)
    print(f"Extracted {len(outputs)} reports successfully.")

    # 2. Convert extractions to Vigipy transaction DataFrame
    print("\n2. Converting extractions to Vigipy transaction format...")
    tx_df = to_vigipy_df(
        outputs,
        records=MAUDE_RECORDS,
        product_key="brand_name",
        target_level="code_term",
        format="transaction",
    )
    print(f"Generated {len(tx_df)} finding transactions. Sample:")
    print(tx_df[["report_id", "product", "category", "finding"]].head(6).to_string(index=False))

    # 3. High-level disproportionality surveillance scan
    try:
        import vigipy
        print("\n3. Executing Disproportionality Surveillance Scan (PRR, ROR, BCPNN, GPS)...")
        results = run_disproportionality_scan(
            outputs=outputs,
            records=MAUDE_RECORDS,
            methods=["prr", "ror", "bcpnn", "gps"],
            min_events=1,
            product_key="brand_name",
            target_level="code_term",
        )

        print("\n--- Proportional Reporting Ratio (PRR) Top Signals ---")
        prr_res = results["prr"]
        print(prr_res.signals[["Product", "Adverse Event", "Count", "Expected Count", "PRR", "p_value"]].head(5).to_string(index=False))

        print("\n--- Reporting Odds Ratio (ROR) Top Signals ---")
        ror_res = results["ror"]
        print(ror_res.signals[["Product", "Adverse Event", "Count", "Expected Count", "ROR", "p_value"]].head(5).to_string(index=False))

        print("\n--- Bayesian Information Component (BCPNN - IC 2.5% Quantile) ---")
        bcpnn_res = results["bcpnn"]
        print(bcpnn_res.all_signals[["Product", "Adverse Event", "Count", "Expected Count", "quantile"]].head(5).to_string(index=False))

        print("\n--- Empirical Bayes Geometric Mean (GPS - EB 5% Log2) ---")
        gps_res = results["gps"]
        print(gps_res.all_signals[["Product", "Adverse Event", "Count", "Expected Count", "log2", "LowerBound"]].head(5).to_string(index=False))

        # 4. Multivariable LASSO Analysis
        print("\n4. Running Multivariable Relaxed LASSO with Sparse Binary Container...")
        bin_container = to_vigipy_container(
            outputs=outputs,
            records=MAUDE_RECORDS,
            binary=True,
            product_key="brand_name",
            target_level="code_term",
            sparse=True,
        )
        lasso_res = vigipy.lasso(
            bin_container,
            min_events=1,
            relaxed=True,
            decision_metric="lower_bound",
            lasso_thresh=0.0,
        )
        print("LASSO Adjusted Odds Ratios (aROR):")
        print(lasso_res.all_signals[["Product", "Adverse Event", "Count", "aROR", "CI Lower", "CI Upper"]].head(5).to_string(index=False))

        print("\n" + "=" * 80)
        print("SUCCESS! Complete workflow verified from raw narratives to signals.")
        print("=" * 80)

    except ImportError:
        print("\nvigipy is not installed in the active environment.")
        print("Install it with: pip install vigipy or pip install spotlight-nlp[vigipy]")


if __name__ == "__main__":
    main()
