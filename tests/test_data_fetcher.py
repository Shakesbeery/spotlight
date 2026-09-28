import json
import io
import urllib.error
from unittest.mock import patch, MagicMock
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


def test_fetch_openfda_pagination_and_api_key():
    """Verifies that requests exceeding 100 records page across offsets,
    and propagate the provided API key."""
    fetcher = MAUDEDataFetcher(data_dir="data", api_key="KEY_INIT_123")

    # Generate mock page payload
    def make_mock_results(start_idx, count):
        results = []
        for i in range(count):
            idx = start_idx + i
            results.append({
                "mdr_report_key": f"KEY_{idx}",
                "event_type": "Malfunction",
                "device": [{"brand_name": f"Device_{idx}", "device_report_product_code": "GAG"}],
                "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": f"Problem narrative {idx}"}]
            })
        return json.dumps({"results": results}).encode("utf-8")

    page_1_data = make_mock_results(0, 100)
    page_2_data = make_mock_results(100, 50)

    mock_resp1 = MagicMock()
    mock_resp1.read.return_value = page_1_data
    mock_resp1.__enter__.return_value = mock_resp1

    mock_resp2 = MagicMock()
    mock_resp2.read.return_value = page_2_data
    mock_resp2.__enter__.return_value = mock_resp2

    with patch("urllib.request.urlopen", side_effect=[mock_resp1, mock_resp2]) as mock_urlopen:
        records = fetcher.fetch_openfda_records(limit=150, api_key="OVERRIDE_KEY_456")

        assert len(records) == 150
        assert records[0].mdr_report_key == "KEY_0"
        assert records[149].mdr_report_key == "KEY_149"
        assert mock_urlopen.call_count == 2

        # Verify query parameters in both calls
        req1 = mock_urlopen.call_args_list[0][0][0]
        req2 = mock_urlopen.call_args_list[1][0][0]
        assert "limit=100" in req1.full_url
        assert "skip=0" in req1.full_url
        assert "api_key=OVERRIDE_KEY_456" in req1.full_url

        assert "limit=50" in req2.full_url
        assert "skip=100" in req2.full_url
        assert "api_key=OVERRIDE_KEY_456" in req2.full_url


def test_fetch_openfda_rate_limit_backoff():
    """Verifies that HTTP 429 rate limit errors trigger backoff and retry."""
    fetcher = MAUDEDataFetcher(data_dir="data")

    page_data = json.dumps({
        "results": [{
            "mdr_report_key": "RETRY_KEY_1",
            "event_type": "Injury",
            "device": [{"brand_name": "TestDev", "device_report_product_code": "LIT"}],
            "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": "Bleeding occurred"}]
        }]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = page_data
    mock_resp.__enter__.return_value = mock_resp

    http_429 = urllib.error.HTTPError(
        url="https://api.fda.gov",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(b"Rate limit exceeded"),
    )

    with patch("urllib.request.urlopen", side_effect=[http_429, mock_resp]) as mock_urlopen, \
         patch("time.sleep") as mock_sleep:
        records = fetcher.fetch_openfda_records(limit=1)

        assert len(records) == 1
        assert records[0].mdr_report_key == "RETRY_KEY_1"
        assert mock_urlopen.call_count == 2
        assert mock_sleep.call_count == 1

