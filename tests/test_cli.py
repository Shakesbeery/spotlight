"""
Unit tests for the Spotlight CLI.
"""

import json
import pytest
from unittest.mock import patch
from spotlight.cli import main, format_extraction_text
from spotlight.pipeline import MAUDEExtractionPipeline


def test_cli_version(capsys):
    with patch("sys.argv", ["spotlight", "version"]):
        main()
    captured = capsys.readouterr()
    assert "Spotlight FDA MAUDE NLP System v" in captured.out


def test_cli_extract_text(capsys):
    text = "Balloon burst under pressure causing arterial dissection."
    with patch("sys.argv", ["spotlight", "extract", text]):
        main()
    captured = capsys.readouterr()
    assert "SPOTLIGHT EXTRACTION SUMMARY" in captured.out
    assert "balloon burst" in captured.out.lower()
    assert "arterial dissection" in captured.out.lower()


def test_cli_extract_json(capsys):
    text = "Balloon burst under pressure."
    with patch("sys.argv", ["spotlight", "extract", text, "--json"]):
        main()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "operational_problems" in data
    assert len(data["operational_problems"]) > 0


def test_cli_batch_file(tmp_path, capsys):
    input_records = [
        "Balloon burst causing dissection.",
        "Dark particulate matter discovered in packaging.",
    ]
    input_file = tmp_path / "test_input.json"
    output_file = tmp_path / "test_output.json"
    input_file.write_text(json.dumps(input_records), encoding="utf-8")

    with patch("sys.argv", ["spotlight", "batch", str(input_file), "-o", str(output_file)]):
        main()

    captured = capsys.readouterr()
    assert "Successfully processed 2 records" in captured.out
    assert output_file.exists()

    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 2
    assert "operational_problems" in saved_data[0]
    assert "manufacturing_issues" in saved_data[1]


def test_cli_mdr_stats_and_query(tmp_path, capsys):
    from spotlight.mdr_database import MDRDatabase
    db_file = tmp_path / "test_cli_mdr.db"
    db = MDRDatabase(db_path=str(db_file))

    # Ingest a mock device and text
    dev_txt = tmp_path / "device.txt"
    dev_txt.write_text("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\n999|TestBrand|GAG\n", encoding="latin-1")
    foi_txt = tmp_path / "foitext.txt"
    foi_txt.write_text("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\n999|D|Stapler jammed.\n", encoding="latin-1")

    db.ingest_txt(str(dev_txt), verbose=False)
    db.ingest_txt(str(foi_txt), verbose=False)

    # Test mdr stats CLI
    with patch("sys.argv", ["spotlight", "mdr", "stats", "--db", str(db_file)]):
        main()
    captured = capsys.readouterr()
    assert "Database Statistics for:" in captured.out
    assert "Device Records:    1" in captured.out

    # Test mdr query CLI
    with patch("sys.argv", ["spotlight", "mdr", "query", "--db", str(db_file), "--product-code", "GAG"]):
        main()
    captured = capsys.readouterr()
    assert "Found 1 records matching query" in captured.out
    assert "TestBrand" in captured.out

    # Test mdr extract CLI
    with patch("sys.argv", ["spotlight", "mdr", "extract", "--db", str(db_file), "--product-code", "GAG"]):
        main()
    captured = capsys.readouterr()
    assert "SPOTLIGHT ORCHESTRATED QUERY-EXTRACT-SAVE PIPELINE" in captured.out
    assert "Pipeline Run Finished" in captured.out

    # Test mdr stats CLI now reports extracted findings
    with patch("sys.argv", ["spotlight", "mdr", "stats", "--db", str(db_file)]):
        main()
    captured = capsys.readouterr()
    assert "Total Extracted Reports: 1" in captured.out
    assert "Total Findings Saved:" in captured.out

    # Test mdr query --unextracted-only reports 0 records now that it was extracted
    with patch("sys.argv", ["spotlight", "mdr", "query", "--db", str(db_file), "--unextracted-only"]):
        main()
    captured = capsys.readouterr()
    assert "Found 0 records matching query (Unextracted Delta Only)" in captured.out

    db.close()


def test_cli_project(tmp_path, capsys):
    from spotlight.mdr_database import MDRDatabase
    raw_db_file = tmp_path / "raw.db"
    raw_db = MDRDatabase(db_path=str(raw_db_file))

    # Ingest mock raw record
    dev_txt = tmp_path / "dev.txt"
    dev_txt.write_text("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\nK100|PacingLead|DTE\n", encoding="latin-1")
    txt_txt = tmp_path / "txt.txt"
    txt_txt.write_text("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\nK100|D|Lead fractured causing cardiac arrest.\n", encoding="latin-1")
    mst_txt = tmp_path / "mst.txt"
    mst_txt.write_text("MDR_REPORT_KEY|EVENT_TYPE\nK100|Injury\n", encoding="latin-1")

    raw_db.ingest_txt(str(dev_txt), verbose=False)
    raw_db.ingest_txt(str(txt_txt), verbose=False)
    raw_db.ingest_txt(str(mst_txt), verbose=False)
    raw_db.close()

    base_dir = tmp_path / "projects"

    # 1. Project create
    with patch("sys.argv", [
        "spotlight", "project", "create", "cardiac_study",
        "--base-dir", str(base_dir),
        "--raw-db", str(raw_db_file),
        "--desc", "Cardiac pacing study",
    ]):
        main()
    captured = capsys.readouterr()
    assert "Project 'cardiac_study' ready." in captured.out

    # 2. Project list
    with patch("sys.argv", ["spotlight", "project", "list", "--base-dir", str(base_dir)]):
        main()
    captured = capsys.readouterr()
    assert "cardiac_study" in captured.out

    # 3. Project extract
    with patch("sys.argv", [
        "spotlight", "project", "extract", "cardiac_study",
        "--base-dir", str(base_dir),
        "--product-code", "DTE",
    ]):
        main()
    captured = capsys.readouterr()
    assert "SPOTLIGHT PROJECT PIPELINE: 'cardiac_study'" in captured.out
    assert "Newly Extracted: 1" in captured.out

    # 4. Project stats
    with patch("sys.argv", ["spotlight", "project", "stats", "cardiac_study", "--base-dir", str(base_dir)]):
        main()
    captured = capsys.readouterr()
    assert "Spotlight Project: cardiac_study" in captured.out
    assert "Total Extracted Rpts: 1" in captured.out

    # 5. Project export
    with patch("sys.argv", [
        "spotlight", "project", "export", "cardiac_study",
        "--base-dir", str(base_dir),
        "-o", "cardiac.csv",
    ]):
        main()
    captured = capsys.readouterr()
    assert "Exported Vigipy dataset to:" in captured.out
    assert (base_dir / "cardiac_study" / "exports" / "cardiac.csv").exists()

