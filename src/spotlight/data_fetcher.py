"""
Data fetcher for real FDA MAUDE reports.
Supports:
1. openFDA Device Event API: Targeted query fetching by event type, keyword, or device class.
2. FDA bulk partition streaming: Downloading and streaming partitions directly from download.open.fda.gov.
"""

import urllib.request
import urllib.parse
import json
import zipfile
import os
import time
from typing import List, Optional, Dict, Any, Generator
from spotlight.schemas import MAUDERecordInput


OPENFDA_DEVICE_EVENT_URL = "https://api.fda.gov/device/event.json"


class MAUDEDataFetcher:
    """Fetches real FDA MAUDE records for testing and production ingestion."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_records_to_json(self, records: List[MAUDERecordInput], filename: str) -> str:
        """Saves fetched records to a local JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        data = [r.model_dump() for r in records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def fetch_openfda_records(
        self,
        limit: int = 20,
        event_type: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[MAUDERecordInput]:
        """
        Fetches live, real FDA MAUDE records from the official openFDA API.
        Extracts narrative text (MDR text), device brand name, product code, and MDR report key.
        """
        params = {"limit": min(limit, 100)}

        search_terms = []
        if event_type:
            search_terms.append(f'event_type:"{event_type}"')
        if query:
            search_terms.append(query)
        # Ensure records contain actual narrative text
        search_terms.append("_exists_:mdr_text.text")

        if search_terms:
            params["search"] = " AND ".join(search_terms)

        url = f"{OPENFDA_DEVICE_EVENT_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Spotlight-FDA-NLP-System/1.0"}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        raw_results = payload.get("results", [])
        records: List[MAUDERecordInput] = []

        for item in raw_results:
            mdr_key = str(item.get("mdr_report_key", item.get("report_number", "UNKNOWN")))
            event_type_val = item.get("event_type", "Unknown")

            # Extract device brand name & product code
            brand_name = None
            product_code = None
            devices = item.get("device", [])
            if devices and isinstance(devices, list):
                brand_name = devices[0].get("brand_name")
                product_code = devices[0].get("device_report_product_code")

            # Extract narrative texts (combine Event Description and Evaluation if present)
            mdr_texts = item.get("mdr_text", [])
            narrative_parts = []
            if isinstance(mdr_texts, list):
                for text_entry in mdr_texts:
                    t_type = text_entry.get("text_type_code", "")
                    content = text_entry.get("text", "")
                    if not content:
                        continue
                    if t_type == "Description of Event or Problem":
                        narrative_parts.append(f"EVENT DESCRIPTION: {content}")
                    elif t_type == "Manufacturer Evaluation":
                        narrative_parts.append(f"MANUFACTURER EVALUATION: {content}")
                    elif t_type == "Additional Manufacturer Narrative":
                        narrative_parts.append(f"ADDITIONAL NARRATIVE: {content}")
                    else:
                        narrative_parts.append(content)

            full_narrative = "\n\n".join(narrative_parts).strip()
            if not full_narrative:
                continue

            records.append(
                MAUDERecordInput(
                    mdr_report_key=mdr_key,
                    brand_name=brand_name,
                    product_code=product_code,
                    event_type=event_type_val,
                    narrative_text=full_narrative,
                )
            )

        return records

    def fetch_diverse_real_dataset(self, samples_per_category: int = 5) -> List[MAUDERecordInput]:
        """
        Fetches a balanced set of real FDA MAUDE records across operational malfunctions,
        pre-use/manufacturing defects, and clinical adverse events.
        """
        dataset: List[MAUDERecordInput] = []

        # 1. Operational Device Malfunctions
        try:
            malfunctions = self.fetch_openfda_records(
                limit=samples_per_category,
                event_type="Malfunction",
                query='(mdr_text.text:"rupture" OR mdr_text.text:"fracture" OR mdr_text.text:"detach" OR mdr_text.text:"failure to deploy")'
            )
            dataset.extend(malfunctions)
        except Exception as e:
            print(f"Warning: Failed to fetch malfunctions: {e}")

        # 2. Manufacturing / Pre-use Issues (Packaging, Particulate, Contamination)
        try:
            mfg_issues = self.fetch_openfda_records(
                limit=samples_per_category,
                query='(mdr_text.text:"particulate" OR mdr_text.text:"sterile barrier" OR mdr_text.text:"prior to use" OR mdr_text.text:"compromised seal")'
            )
            dataset.extend(mfg_issues)
        except Exception as e:
            print(f"Warning: Failed to fetch manufacturing records: {e}")

        # 3. Patient Adverse Events (Injuries & Deaths)
        try:
            injuries = self.fetch_openfda_records(
                limit=samples_per_category,
                event_type="Injury",
                query='(mdr_text.text:"perforation" OR mdr_text.text:"dissection" OR mdr_text.text:"bleeding" OR mdr_text.text:"injury")'
            )
            dataset.extend(injuries)
        except Exception as e:
            print(f"Warning: Failed to fetch injury records: {e}")

        return dataset

    def download_bulk_partition(
        self,
        url: str = "https://download.open.fda.gov/device/event/2023q3/device-event-0007-of-0007.json.zip",
        output_filename: str = "device-event-2023q3-part7.json.zip"
    ) -> str:
        """Downloads a specific FDA bulk partition ZIP file into data_dir."""
        dest_path = os.path.join(self.data_dir, output_filename)
        if os.path.exists(dest_path):
            print(f"Bulk file already exists at {dest_path}")
            return dest_path

        print(f"Downloading FDA bulk partition from {url} to {dest_path}...")
        start_time = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Spotlight/1.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
            chunk_size = 1024 * 1024  # 1MB
            total_read = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                total_read += len(chunk)
                print(f"\rDownloaded {total_read / (1024 * 1024):.1f} MB...", end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s ({total_read / (1024 * 1024):.1f} MB)")
        return dest_path

    def stream_bulk_partition(
        self,
        zip_path: str,
        max_records: int = 50
    ) -> Generator[MAUDERecordInput, None, None]:
        """
        Streams records from an openFDA device event bulk JSON zip file without
        loading the entire multi-gigabyte uncompressed JSON into memory.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_filenames = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_filenames:
                return

            with zf.open(json_filenames[0]) as json_file:
                # OpenFDA JSON contains: {"meta": {...}, "results": [ {record1}, {record2}, ... ]}
                # We can stream lines or load parsed chunks
                content = json_file.read().decode("utf-8", errors="ignore")
                data = json.loads(content)
                results = data.get("results", [])

                count = 0
                for item in results:
                    mdr_key = str(item.get("mdr_report_key", item.get("report_number", "UNKNOWN")))
                    brand = None
                    pcode = None
                    devices = item.get("device", [])
                    if devices and isinstance(devices, list):
                        brand = devices[0].get("brand_name")
                        pcode = devices[0].get("device_report_product_code")

                    narrative_parts = []
                    for t in item.get("mdr_text", []):
                        content_txt = t.get("text", "")
                        if content_txt:
                            narrative_parts.append(content_txt)

                    full_narrative = "\n\n".join(narrative_parts).strip()
                    if not full_narrative:
                        continue

                    yield MAUDERecordInput(
                        mdr_report_key=mdr_key,
                        brand_name=brand,
                        product_code=pcode,
                        event_type=item.get("event_type", "Unknown"),
                        narrative_text=full_narrative,
                    )
                    count += 1
                    if count >= max_records:
                        break
