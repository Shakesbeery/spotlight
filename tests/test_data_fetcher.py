"""
Test MAUDEDataFetcher fetching real FDA records.
"""

from spotlight.data_fetcher import MAUDEDataFetcher


def test_fetch_real_openfda_records():
    fetcher = MAUDEDataFetcher(data_dir="data")
    records = fetcher.fetch_openfda_records(limit=3)

    assert len(records) > 0
    record = records[0]
    assert record.mdr_report_key != ""
    assert len(record.narrative_text) > 0
    print(f"\nFetched Real MDR Key: {record.mdr_report_key}")
    print(f"Device Brand: {record.brand_name}")
    print(f"Narrative snippet: {record.narrative_text[:120]}...")
