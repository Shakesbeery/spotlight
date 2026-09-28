"""
Unit tests for MDR Database module: Catalog resolution, pipe parsing, relational SQLite ingestion,
and streaming query execution.
"""

import os
import tempfile
import pytest
import spotlight
from spotlight.schemas import MAUDERecordInput
from spotlight.mdr_database import (
    MDRCatalog,
    MDRParser,
    MDRDatabase,
    init_mdr_database,
)


def test_mdr_catalog_resolution():
    """Verifies that MDRCatalog resolves canonical FDA URLs across years and file types."""
    # 1. Specific years
    archives_2024 = MDRCatalog.resolve_archives(years=[2024], file_types=["device", "foitext", "mdrfoi"])
    filenames = {a.filename for a in archives_2024}
    assert "device2024.zip" in filenames
    assert "foitext2024.zip" in filenames
    assert any("mdrfoi" in fn for fn in filenames)

    # 2. Recent years
    recent_archives = MDRCatalog.resolve_archives(years="recent")
    assert len(recent_archives) >= 3

    # 3. Canonical URLs format
    for a in archives_2024:
        assert a.url.startswith("https://www.accessdata.fda.gov/MAUDE/ftparea/")
        assert a.url.endswith(".zip")


def test_mdr_parser_and_ingestion():
    """Tests parsing raw pipe-delimited MDR text files and ingesting into SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_maude.db")
        db = MDRDatabase(db_path=db_path)

        # 1. Create sample device.txt file
        device_txt_path = os.path.join(tmpdir, "device2024.txt")
        with open(device_txt_path, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|DEVICE_EVENT_KEY|IMPLANT_FLAG|DATE_REMOVED_FLAG|DEVICE_SEQUENCE_NO|DATE_RECEIVED|BRAND_NAME|GENERIC_NAME|MANUFACTURER_D_NAME|MANUFACTURER_D_ADDRESS_1|MANUFACTURER_D_ADDRESS_2|MANUFACTURER_D_CITY|MANUFACTURER_D_STATE_CODE|MANUFACTURER_D_ZIP_CODE|MANUFACTURER_D_ZIP_CODE_EXT|MANUFACTURER_D_COUNTRY_CODE|MANUFACTURER_D_POSTAL_CODE|DEVICE_OPERATOR|EXPIRATION_DATE_OF_DEVICE|MODEL_NUMBER|CATALOG_NUMBER|LOT_NUMBER|OTHER_ID_NUMBER|DEVICE_AVAILABILITY|DATE_RETURNED_TO_MANUFACTURER|DEVICE_REPORT_PRODUCT_CODE|DEVICE_AGE_TEXT|DEVICE_EVALUATED_BY_MANUFACTUR\n")
            f.write("10001|1|||1|01/15/2024|Stapler Alpha|Surgical Stapler|MedTech Corp|||||||||||MOD-101||LOT-999||||GAG||\n")
            f.write("10002|2|||1|01/20/2024|Catheter Flow|Balloon Dilatation|Vascular Inc|||||||||||MOD-202||LOT-888||||LIT||\n")
            f.write("10003|3|||1|02/05/2024|Stapler Beta|Surgical Stapler|MedTech Corp|||||||||||MOD-102||LOT-777||||GAG||\n")

        # 2. Create sample foitext.txt file (with multiline text and extra pipes)
        text_txt_path = os.path.join(tmpdir, "foitext2024.txt")
        with open(text_txt_path, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|MDR_TEXT_KEY|TEXT_TYPE_CODE|PATIENT_SEQUENCE_NUMBER|DATE_REPORT|FOI_TEXT\n")
            f.write("10001|T1|D|1|01/15/2024|Intraoperatively, the surgical stapler misfired and had a failure to cut. Acute hemorrhage occurred.\n")
            f.write("10001|T2|E|1|01/16/2024|Evaluation: Visual inspection showed jammed anvil mechanism.\n")
            f.write("10002|T3|D|1|01/20/2024|Balloon burst during inflation causing arterial dissection. Covered stent deployed.\n")
            f.write("10003|T4|D|1|02/05/2024|Stapler jammed during firing. Swapped out device. No injury.\n")

        # 3. Create sample mdrfoi.txt file
        master_txt_path = os.path.join(tmpdir, "mdrfoi.txt")
        with open(master_txt_path, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|EVENT_KEY|REPORT_NUMBER|REPORT_SOURCE_CODE|MANUFACTURER_LINK_FLAG|NUMBER_DEVICES_IN_EVENT|NUMBER_PATIENTS_IN_EVENT|DATE_RECEIVED|ADVERSE_EVENT_FLAG|PRODUCT_PROBLEM_FLAG|DATE_REPORT|DATE_OF_EVENT|REPROCESSED_AND_REUSED_FLAG|REPORTER_OCCUPATION_CODE|HEALTH_PROFESSIONAL|INITIAL_REPORT_TO_FDA|DATE_FACILITY_AWARE|REPORT_DATE|EVENT_TYPE|TYPE_OF_REPORT\n")
            f.write("10001|E1|R1|M|Y|1|1|01/15/2024|Y|Y|||||||||Injury|Initial\n")
            f.write("10002|E2|R2|M|Y|1|1|01/20/2024|Y|N|||||||||Injury|Initial\n")
            f.write("10003|E3|R3|M|Y|1|1|02/05/2024|N|Y|||||||||Malfunction|Initial\n")

        # Ingest all files
        rows_dev = db.ingest_txt(device_txt_path, verbose=False)
        rows_txt = db.ingest_txt(text_txt_path, verbose=False)
        rows_mst = db.ingest_txt(master_txt_path, verbose=False)

        assert rows_dev == 3
        assert rows_txt == 4
        assert rows_mst == 3

        # Test statistics
        stats = db.stats()
        assert stats["device_records"] == 3
        assert stats["narrative_chunks"] == 4
        assert stats["master_reports"] == 3
        assert stats["top_product_codes"]["GAG"] == 2
        assert stats["top_product_codes"]["LIT"] == 1
        db.close()


def test_mdr_database_queries_and_streaming():
    """Verifies relational queries and streaming MAUDERecordInput objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_maude.db")
        db = MDRDatabase(db_path=db_path)

        # Setup records
        device_txt = os.path.join(tmpdir, "device.txt")
        with open(device_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\n")
            f.write("REC-1|Stapler Alpha|GAG\n")
            f.write("REC-2|Catheter Flow|LIT\n")
            f.write("REC-3|Stapler Beta|GAG\n")

        text_txt = os.path.join(tmpdir, "foitext.txt")
        with open(text_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\n")
            f.write("REC-1|D|Stapler misfired causing bleeding.\n")
            f.write("REC-1|E|Manufacturer evaluation showed damaged pin.\n")
            f.write("REC-2|D|Catheter balloon ruptured in vivo.\n")
            f.write("REC-3|D|Stapler jammed in jaw.\n")

        master_txt = os.path.join(tmpdir, "mdrfoi.txt")
        with open(master_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|EVENT_TYPE|DATE_RECEIVED\n")
            f.write("REC-1|Injury|2024-01-10\n")
            f.write("REC-2|Injury|2024-01-15\n")
            f.write("REC-3|Malfunction|2024-02-01\n")

        db.ingest_txt(device_txt, verbose=False)
        db.ingest_txt(text_txt, verbose=False)
        db.ingest_txt(master_txt, verbose=False)

        # 1. Query by product code GAG
        gag_records = db.query_records(product_code="GAG")
        assert len(gag_records) == 2
        for r in gag_records:
            assert r.product_code == "GAG"
            assert isinstance(r, MAUDERecordInput)

        # 2. Check multiple narrative chunks are concatenated
        rec_1 = next(r for r in gag_records if r.mdr_report_key == "REC-1")
        assert "Stapler misfired causing bleeding." in rec_1.narrative_text
        assert "Manufacturer evaluation showed damaged pin." in rec_1.narrative_text
        assert rec_1.brand_name == "Stapler Alpha"
        assert rec_1.event_type == "Injury"

        # 3. Query by event type Malfunction
        malfunc_records = db.query_records(event_type="Malfunction")
        assert len(malfunc_records) == 1
        assert malfunc_records[0].mdr_report_key == "REC-3"

        # 4. Stream records with limit
        streamed = list(db.stream_records(limit=2))
        assert len(streamed) == 2

        # 5. Extract findings from streamed records directly with Spotlight!
        extractor = spotlight.get_default_extractor()
        outputs = [extractor.process_record(r) for r in streamed]
        assert len(outputs) == 2
        assert outputs[0].brand_name is not None
        db.close()


def test_init_mdr_database_factory():
    """Verifies factory constructor init_mdr_database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "factory.db")
        db = init_mdr_database(db_path=db_path, sync=False)
        assert isinstance(db, MDRDatabase)
        assert os.path.exists(db_path)
        stats = db.stats()
        assert stats["master_reports"] == 0
        db.close()


def test_event_type_none_and_all_filtering():
    """Verifies that event_type=None, 'all', or '' queries across all event types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_events.db")
        db = MDRDatabase(db_path=db_path)

        dev_txt = os.path.join(tmpdir, "dev.txt")
        with open(dev_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\n1|D1|GAG\n2|D2|GAG\n3|D3|GAG\n")
        txt_txt = os.path.join(tmpdir, "txt.txt")
        with open(txt_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\n1|D|Text 1\n2|D|Text 2\n3|D|Text 3\n")
        mst_txt = os.path.join(tmpdir, "mst.txt")
        with open(mst_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|EVENT_TYPE\n1|Injury\n2|Malfunction\n3|Death\n")

        db.ingest_txt(dev_txt, verbose=False)
        db.ingest_txt(txt_txt, verbose=False)
        db.ingest_txt(mst_txt, verbose=False)

        # None -> returns all 3
        assert db.count_records(event_type=None) == 3
        assert len(db.query_records(event_type=None)) == 3

        # "all" -> returns all 3
        assert db.count_records(event_type="all") == 3
        assert len(db.query_records(event_type="all")) == 3

        # "" -> returns all 3
        assert db.count_records(event_type="") == 3

        # Specific event type -> returns 1
        assert db.count_records(event_type="Injury") == 1
        assert db.count_records(event_type="Malfunction") == 1
        assert db.count_records(event_type="Death") == 1
        db.close()


def test_durable_extraction_store_and_roundtrip():
    """Tests saving and loading extractions with full schema preservation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        db = MDRDatabase(db_path=db_path)

        extractor = spotlight.get_default_extractor()
        rec1 = MAUDERecordInput(
            mdr_report_key="R101",
            brand_name="EndoStaple",
            product_code="GAG",
            event_type="Injury",
            narrative_text="Surgical stapler misfired during bowel resection. Acute bleeding observed. Surgeon performed laparotomy.",
        )
        rec2 = MAUDERecordInput(
            mdr_report_key="R102",
            brand_name="AeroBalloon",
            product_code="LIT",
            event_type="Malfunction",
            narrative_text="Catheter balloon burst during inflation. Dark particulate matter discovered in packaging.",
        )

        out1 = extractor.process_record(rec1)
        out2 = extractor.process_record(rec2)

        # Save batch
        saved_findings = db.save_extractions_batch([out1, out2])
        assert saved_findings > 0

        # Check is_extracted
        assert db.is_extracted("R101") is True
        assert db.is_extracted("R102") is True
        assert db.is_extracted("R999") is False

        # Check get_extracted_keys
        keys = db.get_extracted_keys()
        assert keys == {"R101", "R102"}
        keys_gag = db.get_extracted_keys(product_code="GAG")
        assert keys_gag == {"R101"}

        # Load extractions
        loaded = db.load_extractions()
        assert len(loaded) == 2
        r1 = next(r for r in loaded if r.mdr_report_key == "R101")
        assert r1.brand_name == "EndoStaple"
        assert r1.product_code == "GAG"
        assert r1.event_type == "Injury"
        assert len(r1.operational_problems) > 0
        assert len(r1.clinical_interventions) > 0
        # Check normalized term roundtrip
        op = r1.operational_problems[0]
        assert len(op.normalized_terms) > 0
        assert op.normalized_terms[0].code is not None

        # Check stats
        stats = db.get_extraction_stats()
        assert stats["total_reports_extracted"] == 2
        assert stats["total_findings_saved"] == saved_findings
        assert len(stats["top_imdrf_codes"]) > 0

        db.close()


def test_orchestrate_pipeline_delta_deduplication():
    """Verifies that orchestrate_pipeline processes deltas and skips already-extracted reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_pipeline.db")
        db = MDRDatabase(db_path=db_path)

        dev_txt = os.path.join(tmpdir, "dev.txt")
        with open(dev_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\n")
            f.write("P1|Stapler A|GAG\n")
            f.write("P2|Stapler B|GAG\n")
            f.write("P3|Catheter C|LIT\n")

        txt_txt = os.path.join(tmpdir, "txt.txt")
        with open(txt_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\n")
            f.write("P1|D|Stapler jammed in tissue.\n")
            f.write("P2|D|Stapler blade broken during cut.\n")
            f.write("P3|D|Balloon ruptured under 5 atm.\n")

        mst_txt = os.path.join(tmpdir, "mst.txt")
        with open(mst_txt, "w", encoding="latin-1") as f:
            f.write("MDR_REPORT_KEY|EVENT_TYPE\n")
            f.write("P1|Malfunction\n")
            f.write("P2|Injury\n")
            f.write("P3|Malfunction\n")

        db.ingest_txt(dev_txt, verbose=False)
        db.ingest_txt(txt_txt, verbose=False)
        db.ingest_txt(mst_txt, verbose=False)

        extractor = spotlight.get_default_extractor()

        # Step 1: Run pipeline for first 2 records
        stats1 = db.orchestrate_pipeline(extractor=extractor, limit=2, chunk_size=2, verbose=False)
        assert stats1.total_matched == 3
        assert stats1.already_extracted == 0
        assert stats1.newly_extracted == 2
        assert stats1.total_findings_saved > 0

        # Verify only 1 unextracted record remains
        assert db.count_records(only_unextracted=True) == 1
        unextracted = list(db.stream_records(only_unextracted=True))
        assert len(unextracted) == 1
        assert unextracted[0].mdr_report_key == "P3"

        # Step 2: Run pipeline again without limit; it should extract ONLY the delta record P3
        stats2 = db.orchestrate_pipeline(extractor=extractor, verbose=False)
        assert stats2.total_matched == 3
        assert stats2.already_extracted == 2
        assert stats2.newly_extracted == 1

        # Step 3: Run pipeline a third time: all 3 are extracted, 0 delta
        stats3 = db.orchestrate_pipeline(extractor=extractor, verbose=False)
        assert stats3.total_matched == 3
        assert stats3.already_extracted == 3
        assert stats3.newly_extracted == 0
        assert stats3.total_findings_saved == 0

        # Step 4: Vigipy bridges directly from compiled database
        vdf = db.to_vigipy_df()
        assert len(vdf) > 0
        assert "product" in vdf.columns
        assert "finding" in vdf.columns

        container = db.to_vigipy_container()
        assert container is not None

        db.close()
