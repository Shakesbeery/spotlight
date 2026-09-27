<div align="center">

# Spotlight (spotlight-nlp)

**High-Throughput Clinical NLP Extraction Library for FDA MAUDE Medical Device Narratives**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)]()
[![Throughput](https://img.shields.io/badge/throughput-%7E150%20rec%2Fsec%2Fcore-orange.svg)]()
[![IMDRF Release](https://img.shields.io/badge/IMDRF-2026%20Harmonized-purple.svg)](https://www.imdrf.org/)

</div>

---

## Overview

The United States FDA **MAUDE** (Manufacturer and User Facility Device Experience) database houses over **26 million** adverse event reports describing medical device malfunctions, patient injuries, and deaths. However, the critical regulatory and clinical intelligence—what went wrong, what physical component broke, which clinical injury occurred, and what emergency surgery was required—is buried within unstructured, noisy free-text narratives contaminated with legal boilerplate and redactions (`(b)(4)`, `(b)(6)`).

**Spotlight** is a production-grade, computationally efficient Python library and CLI engineered to ingest FDA MAUDE narratives and output structured, ground-truth clinical extractions across 5 discrete categories:

1. **Operational Problems**: Device failure modes during active use (*IMDRF Annex A*).
2. **Manufacturing Issues**: Pre-use defects, packaging breaches, and foreign particulate (*IMDRF Annex C*).
3. **Patient Adverse Events**: Physiological injury, tissue damage, hemodynamic collapse, and death (*IMDRF Annex E / MedDRA*).
4. **Clinical Interventions**: Rescue surgeries, bail-out stenting, fragment retrievals, blood transfusions, and prolonged anesthesia (*IMDRF Annex F*).
5. **Other Issues**: Shipping damage, transit delays, off-label usage, and user handling errors.

---

## Key Features

- **Cascaded Tiered Architecture**:
  - **Tier 1 Fast-Path**: Sub-millisecond preprocessing, boilerplate stripping, greedy non-overlapping span extraction, and negation masking. Average CPU latency of **5–15 ms per record** (~150 records/sec/core).
  - **Tier 2 Edge SLM Fallback**: Targeted invocation of **Gemma 4** architectures with strict JSON schema constraints for ambiguous or high-entropy edge cases.
  - **Frontier LLM Distillation**: Integrated harness for **Gemini 3.8 Flash** for synthetic gold dataset generation and model distillation.
- **Granular Regulatory Grounding (IMDRF 2026 & MedDRA)**:
  - Eliminates ontology collapse by providing Level 3/4 code distinctions (e.g. separating complete fractures `A040101` from surface cracks `A0404`, balloon bursts `A0414`, misfires `A050502`, and cutting failures `A050702`).
- **Anatomical Component Linkage (IMDRF Annex G)**:
  - Syntactically binds physical device components (e.g. `sheath hub`, `polyaxial tulip collar`, `hemostasis valve junction`) directly to their specific failure predicate.
- **Entity Resolution & Deduplication**:
  - Resolves acronym coreferences (e.g., `coronary artery bypass graft` vs. `CABG`) while preserving multi-aspect physical manifestations that share parent ontology branches.
- **Direct openFDA & Bulk Streaming**:
  - Out-of-the-box support for querying the official openFDA API and streaming multi-gigabyte historical quarterly partitions directly from `download.open.fda.gov` without memory exhaustion.
- **Zero Heavyweight Runtime Dependencies**:
  - Core library runs on pure Python and Pydantic v2—no mandatory PyTorch or heavy C++ binaries required for standard high-throughput extraction.

---

## Installation

### From Source or Git Repository
```bash
git clone https://github.com/spotlight-nlp/spotlight.git
cd spotlight
pip install .
```

### For Development & Testing
```bash
pip install -e ".[dev]"
```

### With Optional LLM Fallback Dependencies
```bash
pip install -e ".[llm]"
```

---

## Quickstart (Python API)

### 1. One-Liner Narrative Extraction
```python
import spotlight

narrative = (
    "During laparoscopic colectomy, the surgical stapler misfired across the mesenteric vessels "
    "and demonstrated complete failure to cut. The jammed jaw caused severe tissue laceration and "
    "acute hemorrhage. The surgical team performed an emergent exploratory laparotomy to achieve "
    "hemostasis. The patient received a blood transfusion of 2 units of PRBC and required prolonged anesthesia time."
)

result = spotlight.extract(narrative, brand_name="Articulating Stapler")

# Access structured operational problems
for problem in result.operational_problems:
    print(f"[{problem.id}] {problem.verbatim_span} | Component: {problem.affected_component}")
    for term in problem.normalized_terms:
        print(f"     -> IMDRF Annex A: {term.code} ({term.preferred_term})")

# Access adverse events
for ae in result.adverse_events:
    print(f"[{ae.id}] {ae.verbatim_span}")
    for term in ae.normalized_terms:
        print(f"     -> IMDRF Annex E: {term.code} ({term.preferred_term})")

# Access clinical rescue interventions
for intv in result.clinical_interventions:
    print(f"[{intv.id}] {intv.verbatim_span} (Triggered by: {intv.triggered_by})")
    for term in intv.normalized_terms:
        print(f"     -> IMDRF Annex F: {term.code} ({term.preferred_term})")
```

### 2. Batch Processing with High Throughput
```python
import spotlight

narratives = [
    "Prior to surgery, visible dark particulate matter was identified adhering to the implant.",
    "During PCI, the balloon burst under 14 atm causing acute coronary dissection. Emergency covered stent placed.",
    "Catheter shaft fractured during advancement. Snare catheter retrieval performed to extract fragment.",
    "Routine infusion completed with no patient injury and no adverse events.",
]

# Processes records sequentially or concurrently
results = spotlight.extract_batch(narratives)

for i, res in enumerate(results, 1):
    print(f"Report {i}: Processed in {res.processing_time_ms:.2f} ms via {res.execution_path.value}")
```

### 3. Fetching Real-World Data from openFDA
```python
from spotlight import MAUDEDataFetcher, SpotlightExtractor

fetcher = MAUDEDataFetcher(data_dir="data")

# Queries live openFDA Device Event API
records = fetcher.fetch_openfda_records(
    limit=10,
    event_type="Injury",
    query='(mdr_text.text:"perforation" OR mdr_text.text:"dissection")'
)

extractor = SpotlightExtractor()
outputs = extractor.process_batch(records)

print(f"Successfully processed {len(outputs)} real FDA records.")
```

---

## Command-Line Interface (CLI)

Spotlight includes a complete CLI:

### Analyze a Single Narrative
```bash
spotlight extract "During catheterization, the balloon burst under pressure causing arterial dissection."
```

### Output JSON to stdout or file
```bash
spotlight extract "Balloon burst under pressure." --json
```

### Batch Ingest a File of Records
```bash
spotlight batch input_records.json -o extracted_findings.json
```

### Fetch Live FDA Reports
```bash
spotlight fetch --limit 20 --event-type Malfunction -o live_malfunctions.json
```

### Check Installed Version
```bash
spotlight version
```

---

## Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Raw FDA MAUDE Narrative             │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Stage 1: Preprocessor               │
                  │  • Strip CFR 803 & manufacturer disclaimers  │
                  │  • Remove (b)(4) & (b)(6) redaction tokens   │
                  │  • Segment clauses & tag temporal timing     │
                  │  • Filter negated clinical contexts          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Tier 1: High-Speed Greedy Span Triage     │
                  │  • Non-overlapping longest span matching     │
                  │  • Sub-millisecond CPU execution             │
                  └──────────────┬───────────────────────────────┘
                                 │
                     Confidence < 0.80 or Ambiguous?
                                ╱ ╲
                               YES NO
                              ╱     ╲
                             ▼       ▼
    ┌──────────────────────────────┐  │
    │   Tier 2: Gemma 4 SLM        │  │
    │   Structured JSON extraction │  │
    └──────────────┬───────────────┘  │
                   │                  │
                   └─────────┬────────┘
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │          IMDRF & MedDRA Ontology Grounding Engine          │
    │  • Annex A (Device Problems)   • Annex C (Pre-Use / Mfg)   │
    │  • Annex E (Adverse Events)    • Annex F (Interventions)   │
    └────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │     Component Linker & Entity Deduplication Resolver       │
    │  • Bind physical components (sheath hub, collar, jaw)      │
    │  • Resolve acronym coreference (CABG <-> bypass graft)     │
    │  • Attribute clinical interventions to adverse events      │
    └────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │       Structured MAUDEExtractionOutput (JSON / Object)     │
    └────────────────────────────────────────────────────────────┘
```

---

## Regulatory Ontology Grounding Reference

Spotlight maps free-text clinical and engineering mentions directly into the latest **2026 IMDRF Harmonized Adverse Event Terminology**:

| Category | Example Mentions | IMDRF Code | Preferred Term |
| :--- | :--- | :--- | :--- |
| **Operational Problem** | `"hub was fractured"`, `"collar broke"` | **`A040101`** | Fracture |
| **Operational Problem** | `"hub cracked"`, `"hairline crack"` | **`A0404`** | Crack |
| **Operational Problem** | `"balloon burst under pressure"` | **`A0414`** | Material Split, Cut or Torn |
| **Operational Problem** | `"detached into epidural space"` | **`A0501`** | Detachment of Device or Device Component |
| **Operational Problem** | `"stent dislodged from balloon"` | **`A051201`** | Device Dislodged or Dislocated |
| **Operational Problem** | `"misfired across vessels"` | **`A050502`** | Misfire |
| **Operational Problem** | `"failure to cut"` | **`A050702`** | Failure to Cut |
| **Operational Problem** | `"stapler jammed"` | **`A0506`** | Mechanical Jam |
| **Manufacturing Issue** | `"visible dark particulate matter"` | **`C1901`** | Foreign Material / Particulate |
| **Manufacturing Issue** | `"crack in inner sterile barrier tray"` | **`C0202`** | Seal Integrity / Sterile Barrier Compromised |
| **Manufacturing Issue** | `"sealing cap was loose"` | **`C0102`** | Loose / Disassembled Component |
| **Adverse Event** | `"mesenteric tissue laceration"` | **`E2101`** | Tissue Laceration |
| **Adverse Event** | `"unintended dural tear"` | **`E2103`** | Tissue / Dural Tear |
| **Adverse Event** | `"vessel perforation"` | **`E2114`** | Perforation |
| **Adverse Event** | `"acute hemorrhage"`, `"severe bleeding"` | **`E0506`** | Hemorrhage / Bleeding |
| **Adverse Event** | `"sudden hemodynamic instability"` | **`E0401`** | Hypotension / Shock / Hemodynamic Instability |
| **Adverse Event** | `"acute coronary dissection"` | **`E2401`** | Vascular Dissection |
| **Clinical Intervention**| `"snare catheter retrieval of fragment"`| **`F1903`** | Device Explantation / Fragment Retrieval |
| **Clinical Intervention**| `"coronary artery bypass graft (CABG)"` | **`F1901`** | Surgical Revision / Revascularization |
| **Clinical Intervention**| `"emergency exploratory laparotomy"` | **`F19`** | Surgical Intervention |
| **Clinical Intervention**| `"emergency covered stent"` | **`F1905`** | Rescue / Bailout Percutaneous Stenting |
| **Clinical Intervention**| `"transfusion of 2 units of PRBC"` | **`F2302`** | Blood Product Transfusion |
| **Clinical Intervention**| `"prolonged anesthesia time"` | **`F1908`** | Prolonged Surgery / Extended Procedure |
| **Clinical Intervention**| `"fluid resuscitation & vasopressors"` | **`F2301`** | Pharmacological / Medical Support |

---

## Verification & Benchmarks

The entire test suite can be run via `pytest`:

```bash
pytest tests/ -v
```

### Benchmark Summary (Single CPU Core)
- **Fast-Path Latency**: **5.8 ms – 19.4 ms** per narrative.
- **Throughput**: **100 – 150 records / second / core**.
- **Memory Overhead**: < 150 MB peak resident set size.
- **Test Suite**: **43 passing tests** covering preprocessing, token triage, component linkage, acronym reconciliation, ontology disambiguation, API ergonomics, and CLI commands.

---

## Open Source License & Compliance Review

- **Library License**: Distributed under the permissive **Apache License, Version 2.0** (`LICENSE`). Free for both commercial and academic integration.
- **Dependencies**: All core dependencies (`pydantic`, `pytest`, `pluggy`, etc.) use permissive OSI-approved licenses (MIT, BSD-3, PSF). There are **zero GPL / AGPL copyleft dependencies**.
- **Public Domain Data**:
  - FDA MAUDE data and openFDA APIs are works of the United States Government under 17 U.S.C. § 105 and reside in the public domain.
  - IMDRF adverse event terminologies are harmonized international regulatory standards published freely for post-market surveillance.

---

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on GitHub.

```bash
git checkout -b feature/my-feature
pytest tests/ -v
git commit -m "Add feature"
```
