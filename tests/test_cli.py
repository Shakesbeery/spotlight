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
