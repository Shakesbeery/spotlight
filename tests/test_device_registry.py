"""
Unit and integration tests for the optional FDA Device & Manufacturer Registry Linker.
Verifies the 4-tier waterfall resolution engine against deterministic and heuristic keys.
"""

import pytest
from spotlight.device_registry import (
    DeviceRegistryLinker,
    MatchTier,
    RegisteredManufacturer,
    RegisteredDevice,
    DeviceResolutionResult,
)
from spotlight.mdr_database import MDRDatabase


@pytest.fixture
def seeded_linker():
    """Provides an in-memory DeviceRegistryLinker pre-seeded with representative FDA data."""
    linker = DeviceRegistryLinker(db_path=":memory:")
    linker.seed_mock_registry()
    return linker


def test_seeded_registry_stats(seeded_linker):
    """Verifies that the mock registry pre-seeds establishments, submissions, and UDIs."""
    stats = seeded_linker.stats()
    assert stats["establishments"] >= 5
    assert stats["premarket_submissions"] >= 6
    assert stats["gudid_records"] >= 4
    assert stats["device_listings"] >= 5


def test_tier_1_udi_resolution(seeded_linker):
    """
    Tier 1: Reports carrying an official UDI-DI must affirmatively resolve to AccessGUDID
    with confidence 1.0.
    """
    res = seeded_linker.resolve_report(
        mdr_report_key="R-UDI-001",
        udi_di="00884521034812",
        brand_name="Reported Stapler",
    )

    assert res.match_tier == MatchTier.TIER_1_UDI
    assert res.is_affirmative is True
    assert res.confidence_score == 1.0
    assert res.device is not None
    assert res.device.proprietary_name == "ECHELON FLEX 60"
    assert res.device.listing_number == "D201452"
    assert res.manufacturer is not None
    assert res.manufacturer.name == "Ethicon Endo-Surgery, LLC"
    assert res.manufacturer.source_table == "gudid"


def test_tier_2_premarket_clearance_510k_and_pma(seeded_linker):
    """
    Tier 2: Reports carrying a 510(k) or PMA number must affirmatively resolve
    to the FDA Premarket database with confidence 1.0.
    """
    # 510(k) test
    res_k = seeded_linker.resolve_report(
        mdr_report_key="R-PMN-001",
        pma_pmn_num="K201452",
        brand_name="Echelon Stapler",
    )
    assert res_k.match_tier == MatchTier.TIER_2_PREMARKET
    assert res_k.is_affirmative is True
    assert res_k.confidence_score == 1.0
    assert res_k.device is not None
    assert res_k.device.premarket_number == "K201452"
    assert res_k.device.premarket_type == "510(k)"
    assert res_k.device.applicant_or_labeler == "Ethicon Endo-Surgery, LLC"
    # Manufacturer enriched from establishment table
    assert res_k.manufacturer is not None
    assert res_k.manufacturer.registration_number == "1820334"
    assert res_k.manufacturer.state == "OH"

    # PMA test
    res_p = seeded_linker.resolve_report(
        mdr_report_key="R-PMA-001",
        pma_pmn_num="P160002",
        brand_name="Coronary Stent",
    )
    assert res_p.match_tier == MatchTier.TIER_2_PREMARKET
    assert res_p.is_affirmative is True
    assert res_p.device.premarket_type == "PMA"
    assert "Resolute Onyx" in res_p.device.proprietary_name
    assert res_p.manufacturer.name == "Medtronic Vascular"


def test_tier_3_report_number_establishment_registration(seeded_linker):
    """
    Tier 3: Mandatory manufacturer reports adhering to 21 CFR § 803.52 3-part syntax
    must yield the manufacturer's FDA Establishment Registration Number / FEI.
    """
    res = seeded_linker.resolve_report(
        mdr_report_key="R-MFR-001",
        report_number="2183427-2024-00192",  # Medtronic Vascular FEI/Reg No
        brand_name="Unidentified Catheter",
        product_code="NIQ",
    )
    assert res.match_tier == MatchTier.TIER_3_REPORT_NUMBER
    assert res.is_affirmative is True
    assert res.confidence_score == 1.0
    assert res.manufacturer is not None
    assert res.manufacturer.registration_number == "2183427"
    assert res.manufacturer.name == "Medtronic Vascular"
    assert res.manufacturer.city == "Santa Rosa"
    assert res.manufacturer.state == "CA"


def test_tier_4_heuristic_fallback(seeded_linker):
    """
    Tier 4: Reports lacking UDI, PMA, or manufacturer report number fall back to
    product code constraint + token similarity matching against listed proprietary names.
    """
    res = seeded_linker.resolve_report(
        mdr_report_key="R-HEUR-001",
        report_number="MW5012345",  # Voluntary MedWatch tracking number
        brand_name="Echelon Flex Endopath Powered",
        product_code="GAG",
    )
    assert res.match_tier == MatchTier.TIER_4_HEURISTIC
    assert res.is_affirmative is False  # Heuristic matches are non-affirmative
    assert res.confidence_score >= 0.40
    assert res.device is not None
    assert res.device.product_code == "GAG"
    assert res.device.proprietary_name == "Echelon Flex Endopath"
    assert res.manufacturer is not None
    assert "Ethicon" in res.manufacturer.name


def test_unresolved_voluntary_report(seeded_linker):
    """
    Voluntary report with ambiguous brand and no regulatory keys must safely return UNRESOLVED.
    """
    res = seeded_linker.resolve_report(
        mdr_report_key="R-UNRES-001",
        report_number="MW9999999",
        brand_name="Generic Unknown Device",
        product_code="ZZZ",
    )
    assert res.match_tier == MatchTier.UNRESOLVED
    assert res.is_affirmative is False
    assert res.confidence_score == 0.0
    assert res.manufacturer is None
    assert res.device is None


def test_custom_file_ingestion(tmp_path):
    """Verifies ingesting custom pipe-delimited establishment and 510(k) flat files."""
    linker = DeviceRegistryLinker(db_path=":memory:")

    # Ingest custom establishment file
    est_file = tmp_path / "establishment.txt"
    est_file.write_text(
        "REGISTRATION_NUMBER|FEI_NUMBER|ESTABLISHMENT_NAME|ADDRESS|CITY|STATE_CODE|ZIP_CODE|COUNTRY_CODE\n"
        "9988776|9988776|Acme Surgical Implants|100 Industrial Pkwy|Austin|TX|78701|US\n",
        encoding="latin-1",
    )
    count_est = linker.ingest_establishment_file(str(est_file))
    assert count_est == 1

    # Ingest custom 510(k) file
    pmn_file = tmp_path / "pmn.txt"
    pmn_file.write_text(
        "KNUMBER|APPLICANT|DEVICENAME|PRODUCTCODE|DATECOMPLETED\n"
        "K240001|Acme Surgical Implants|Acme Titanium Bone Plate|HRS|2024-03-01\n",
        encoding="latin-1",
    )
    count_pmn = linker.ingest_pmn_file(str(pmn_file))
    assert count_pmn == 1

    # Resolve report against newly ingested custom records
    res = linker.resolve_report(
        mdr_report_key="R-ACME-1",
        pma_pmn_num="K240001",
    )
    assert res.match_tier == MatchTier.TIER_2_PREMARKET
    assert res.is_affirmative is True
    assert res.device.proprietary_name == "Acme Titanium Bone Plate"
    assert res.manufacturer.name == "Acme Surgical Implants"
    assert res.manufacturer.state == "TX"


def test_mdr_database_linking_integration(tmp_path, seeded_linker):
    """
    Verifies linking an entire local MDRDatabase in batch, persisting results
    into report_registry_links table.
    """
    raw_db_path = tmp_path / "test_link.db"
    db = MDRDatabase(db_path=str(raw_db_path))

    # Ingest 3 mock records:
    # 1. Has UDI
    # 2. Has PMA
    # 3. Has Manufacturer Report Number
    dev_txt = tmp_path / "dev.txt"
    dev_txt.write_text(
        "MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE|PMA_PMN_NUM|UDI-DI\n"
        "M1|Stapler A|GAG||00884521034812\n"
        "M2|Stent B|NIQ|P160002|\n"
        "M3|Balloon C|LIT||\n",
        encoding="latin-1",
    )
    txt_txt = tmp_path / "txt.txt"
    txt_txt.write_text(
        "MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\n"
        "M1|D|Stapler fired.\n"
        "M2|D|Stent placed.\n"
        "M3|D|Balloon dilated.\n",
        encoding="latin-1",
    )
    mst_txt = tmp_path / "mst.txt"
    mst_txt.write_text(
        "MDR_REPORT_KEY|REPORT_NUMBER|EVENT_TYPE\n"
        "M1|1820334-2024-00001|Malfunction\n"
        "M2|2183427-2024-00002|Injury\n"
        "M3|1002019-2024-00003|Malfunction\n",
        encoding="latin-1",
    )

    db.ingest_txt(str(dev_txt), verbose=False)
    db.ingest_txt(str(txt_txt), verbose=False)
    db.ingest_txt(str(mst_txt), verbose=False)

    # Execute database linking
    stats = seeded_linker.link_mdr_database(db, verbose=False)
    assert stats["total_records_analyzed"] == 3
    assert stats["affirmative_matches"] == 3
    assert stats["tier_breakdown"][MatchTier.TIER_1_UDI.value] == 1
    assert stats["tier_breakdown"][MatchTier.TIER_2_PREMARKET.value] == 1
    assert stats["tier_breakdown"][MatchTier.TIER_3_REPORT_NUMBER.value] == 1

    # Verify rows written to report_registry_links
    conn = db._get_connection()
    links = conn.execute("SELECT mdr_report_key, match_tier, is_affirmative, manufacturer_name FROM report_registry_links ORDER BY mdr_report_key").fetchall()
    conn.close()
    db.close()

    assert len(links) == 3
    assert links[0][0] == "M1"
    assert links[0][1] == MatchTier.TIER_1_UDI.value
    assert links[0][2] == 1
    assert "Ethicon" in links[0][3]

    assert links[1][0] == "M2"
    assert links[1][1] == MatchTier.TIER_2_PREMARKET.value
    assert "Medtronic" in links[1][3]

    assert links[2][0] == "M3"
    assert links[2][1] == MatchTier.TIER_3_REPORT_NUMBER.value
    assert "Boston Scientific" in links[2][3]
