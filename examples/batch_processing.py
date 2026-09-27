"""
Spotlight Batch Processing Example.
Demonstrates high-throughput batch extraction across multiple reports.
"""

import time
import spotlight
from spotlight import MAUDERecordInput


def main():
    narratives = [
        "Prior to surgery, dark particulate matter was discovered adhering to the implant. The sterile tray was cracked.",
        "During PCI, the balloon burst under 14 atm of pressure, causing coronary dissection. Required emergency covered stent.",
        "The catheter shaft was fractured during advancement. Snare catheter retrieval performed to extract fragment.",
        "Routine infusion completed with no patient injury, no adverse events, and no device malfunction.",
        "The polyaxial collar broke off the pedicle screw and detached into the epidural space causing a dural tear.",
    ]

    print(f"Processing batch of {len(narratives)} narratives...")
    start = time.perf_counter()
    results = spotlight.extract_batch(narratives)
    elapsed_total_ms = (time.perf_counter() - start) * 1000.0

    print("\n" + "=" * 70)
    print(f"BATCH PROCESSED : {len(results)} records")
    print(f"TOTAL TIME      : {elapsed_total_ms:.2f} ms")
    print(f"AVG PER RECORD  : {elapsed_total_ms / len(results):.2f} ms")
    print(f"THROUGHPUT      : {len(results) / (elapsed_total_ms / 1000.0):.1f} records/sec")
    print("=" * 70)

    for i, res in enumerate(results, 1):
        findings_count = (
            len(res.operational_problems)
            + len(res.manufacturing_issues)
            + len(res.adverse_events)
            + len(res.clinical_interventions)
        )
        print(f"Record {i}: {findings_count} findings extracted in {res.processing_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
