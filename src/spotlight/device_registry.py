"""
Spotlight FDA Device & Manufacturer Registry Linker.

Provides an optional, high-performance waterfall resolution engine to affirmatively
associate FDA MAUDE/MDR adverse event reports with officially registered devices,
premarket clearances/approvals, and registered manufacturing establishments.

Waterfall Matching Hierarchy:
-----------------------------
1. Tier 1 (UDI-DI): Deterministic 1-to-1 match against FDA AccessGUDID (Global UDI Database).
2. Tier 2 (Premarket Number): Deterministic 1-to-1 match against FDA 510(k) (K######),
   PMA (P######), De Novo (DEN######), or HDE (H######) registries.
3. Tier 3 (Report Number Registration/FEI): Deterministic extraction of the manufacturer's
   7-to-10 digit FDA Establishment Registration Number / FEI from mandatory 3-segment
   report numbers (under 21 CFR § 803.52: [RegNo]-[Year]-[SeqNo]).
4. Tier 4 (Product Code + Normalized Brand Heuristic): Constrained fuzzy/token similarity
   fallback for voluntary (MedWatch 3500) and hospital (3500A) reports.
"""

import os
import re
import csv
import sqlite3
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Sequence, Tuple, Union, Set
from pydantic import BaseModel, Field, ConfigDict


class MatchTier(str, Enum):
    """Hierarchy tier of the device/manufacturer association."""
    TIER_1_UDI = "tier_1_udi"                    # 100% deterministic: GUDID Barcode DI
    TIER_2_PREMARKET = "tier_2_premarket"        # 100% deterministic: 510(k) / PMA clearance
    TIER_3_REPORT_NUMBER = "tier_3_report_number"# 100% deterministic: 21 CFR 803 3-part report number
    TIER_4_HEURISTIC = "tier_4_heuristic"        # Probabilistic: Product code + brand similarity
    UNRESOLVED = "unresolved"                    # No verifiable association found


class RegisteredManufacturer(BaseModel):
    """Officially registered medical device manufacturer or establishment."""
    registration_number: Optional[str] = Field(None, description="FDA Establishment Registration Number (7-10 digits)")
    fei_number: Optional[str] = Field(None, description="FDA Establishment Identifier (FEI)")
    owner_operator_number: Optional[str] = Field(None, description="FDA Owner/Operator Number")
    name: str = Field(..., description="Official legal firm or establishment name")
    address: Optional[str] = Field(None, description="Physical facility street address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="US State or territory code")
    zip_code: Optional[str] = Field(None, description="Postal / ZIP code")
    country: Optional[str] = Field("US", description="Country code (ISO or FDA 2-letter)")
    source_table: str = Field("establishment", description="Registry table providing this record")

    model_config = ConfigDict(frozen=True)


class RegisteredDevice(BaseModel):
    """Officially cleared, approved, or listed medical device."""
    premarket_number: Optional[str] = Field(None, description="Premarket clearance number (K######, P######, DEN######)")
    premarket_type: Optional[str] = Field(None, description="Submission type: 510(k), PMA, De Novo, HDE, Exempt")
    udi_di: Optional[str] = Field(None, description="Global Unique Device Identifier - Device Identifier (GTIN / HIBC)")
    listing_number: Optional[str] = Field(None, description="FDA Device Listing Number (D######)")
    proprietary_name: Optional[str] = Field(None, description="Official cleared or listed commercial trade name")
    generic_name: Optional[str] = Field(None, description="Regulation generic nomenclature")
    product_code: Optional[str] = Field(None, description="FDA 3-letter classification product code (e.g. GAG)")
    applicant_or_labeler: Optional[str] = Field(None, description="Legal premarket applicant or GUDID labeler company")
    decision_date: Optional[str] = Field(None, description="FDA approval/clearance date (YYYY-MM-DD)")

    model_config = ConfigDict(frozen=True)


class DeviceResolutionResult(BaseModel):
    """Outcome of resolving an MDR report against registered device & manufacturer databases."""
    mdr_report_key: str = Field(..., description="Target MDR Report Key")
    match_tier: MatchTier = Field(..., description="Waterfall tier that resolved the entity")
    is_affirmative: bool = Field(..., description="True if matched via deterministic regulatory keys (Tiers 1-3)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Match confidence [0.0 - 1.0]")
    manufacturer: Optional[RegisteredManufacturer] = Field(None, description="Resolved registered manufacturer entity")
    device: Optional[RegisteredDevice] = Field(None, description="Resolved registered device entity")
    input_keys: Dict[str, Any] = Field(default_factory=dict, description="Input parameters supplied to the resolver")
    notes: str = Field("", description="Resolution rationale, parsed tokens, or diagnostic messages")

    model_config = ConfigDict(frozen=True)


class DeviceRegistryLinker:
    """
    Offline/Online Waterfall Resolver linking raw MAUDE reports to FDA registrations.

    Uses an embedded or persistent SQLite reference store indexed on:
    - GUDID Device Identifiers (UDI-DI)
    - 510(k) and PMA numbers (PMA_PMN_NUM)
    - FDA Establishment Registration numbers and FEIs
    - FDA Device Listings by Product Code
    """

    RE_PREMARKET = re.compile(r"\b([KkPpDd][0-9]{6}|[Dd][Ee][Nn][0-9]{6}|[Hh][0-9]{6})\b")
    RE_MFR_REPORT_NUM = re.compile(r"^\s*([0-9]{7,10})\s*-\s*([0-9]{4})\s*-\s*([0-9]{1,6})\s*$")
    RE_UDI_CLEAN = re.compile(r"[^A-Za-z0-9]")

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the registry linker.
        If db_path is None or ":memory:", operates as a fast in-memory SQLite database.
        """
        self.db_path = db_path or ":memory:"
        self._shared_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -32000")
        return conn

    def _release_connection(self, conn: sqlite3.Connection):
        if self._shared_conn is None:
            conn.close()

    def close(self):
        """Closes any open in-memory or persistent connection."""
        if self._shared_conn is not None:
            try:
                self._shared_conn.close()
            except Exception:
                pass
            self._shared_conn = None

    def _init_schema(self):
        """Creates reference schema for official FDA registries."""
        conn = self._get_connection()
        try:
            # 1. Official FDA Establishment Registrations (from foirl/establishment.txt)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ref_establishment (
                    registration_number TEXT PRIMARY KEY,
                    fei_number TEXT,
                    owner_operator_number TEXT,
                    firm_name TEXT NOT NULL,
                    address TEXT,
                    city TEXT,
                    state TEXT,
                    zip_code TEXT,
                    country TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_est_fei ON ref_establishment(fei_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_est_name ON ref_establishment(firm_name)")

            # 2. Premarket Submissions (510(k), PMA, De Novo, HDE)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ref_pmarket (
                    premarket_number TEXT PRIMARY KEY,
                    premarket_type TEXT,
                    applicant_name TEXT,
                    device_name TEXT,
                    product_code TEXT,
                    decision_date TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_pm_pcode ON ref_pmarket(product_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_pm_name ON ref_pmarket(device_name)")

            # 3. AccessGUDID Unique Device Identifiers (UDI-DI)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ref_gudid (
                    udi_di TEXT PRIMARY KEY,
                    brand_name TEXT,
                    version_model TEXT,
                    company_name TEXT,
                    listing_number TEXT,
                    product_code TEXT,
                    duns_number TEXT,
                    fei_number TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_gudid_brand ON ref_gudid(brand_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_gudid_pcode ON ref_gudid(product_code)")

            # 4. Device Listings (from foirl/device_listing.txt)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ref_listing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_number TEXT,
                    product_code TEXT,
                    proprietary_name TEXT,
                    firm_name TEXT,
                    pmn_number TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_list_pcode ON ref_listing(product_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ref_list_name ON ref_listing(proprietary_name)")

            conn.commit()
        finally:
            self._release_connection(conn)

    # =========================================================================
    # INGESTION APIS FOR OFFICIAL FDA REFERENCE DATASETS
    # =========================================================================

    def add_establishment(
        self,
        registration_number: str,
        firm_name: str,
        fei_number: Optional[str] = None,
        owner_operator_number: Optional[str] = None,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zip_code: Optional[str] = None,
        country: Optional[str] = "US",
    ):
        """Manually registers an official FDA establishment."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ref_establishment
                (registration_number, fei_number, owner_operator_number, firm_name, address, city, state, zip_code, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(registration_number).strip(),
                str(fei_number).strip() if fei_number else None,
                str(owner_operator_number).strip() if owner_operator_number else None,
                firm_name.strip(),
                address, city, state, zip_code, country
            ))
            conn.commit()
        finally:
            self._release_connection(conn)

    def add_premarket_device(
        self,
        premarket_number: str,
        applicant_name: str,
        device_name: str,
        product_code: Optional[str] = None,
        premarket_type: Optional[str] = None,
        decision_date: Optional[str] = None,
    ):
        """Manually registers an official cleared/approved 510(k), PMA, or De Novo submission."""
        p_clean = premarket_number.strip().upper()
        if not premarket_type:
            if p_clean.startswith("K"):
                premarket_type = "510(k)"
            elif p_clean.startswith("P"):
                premarket_type = "PMA"
            elif p_clean.startswith("DEN"):
                premarket_type = "De Novo"
            elif p_clean.startswith("H"):
                premarket_type = "HDE"
            else:
                premarket_type = "Premarket"

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ref_pmarket
                (premarket_number, premarket_type, applicant_name, device_name, product_code, decision_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (p_clean, premarket_type, applicant_name.strip(), device_name.strip(), product_code, decision_date))
            conn.commit()
        finally:
            self._release_connection(conn)

    def add_gudid_device(
        self,
        udi_di: str,
        brand_name: str,
        company_name: str,
        version_model: Optional[str] = None,
        listing_number: Optional[str] = None,
        product_code: Optional[str] = None,
        duns_number: Optional[str] = None,
        fei_number: Optional[str] = None,
    ):
        """Manually registers an AccessGUDID device identifier."""
        u_clean = self._clean_udi(udi_di)
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ref_gudid
                (udi_di, brand_name, version_model, company_name, listing_number, product_code, duns_number, fei_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (u_clean, brand_name.strip(), version_model, company_name.strip(), listing_number, product_code, duns_number, fei_number))
            conn.commit()
        finally:
            self._release_connection(conn)

    def add_device_listing(
        self,
        listing_number: str,
        proprietary_name: str,
        product_code: str,
        firm_name: Optional[str] = None,
        pmn_number: Optional[str] = None,
    ):
        """Manually registers an FDA device listing."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO ref_listing
                (listing_number, product_code, proprietary_name, firm_name, pmn_number)
                VALUES (?, ?, ?, ?, ?)
            """, (listing_number.strip(), product_code.strip().upper(), proprietary_name.strip(), firm_name, pmn_number))
            conn.commit()
        finally:
            self._release_connection(conn)

    def seed_mock_registry(self):
        """
        Pre-seeds a rich, realistic sample of major FDA registrations across
        interventional cardiology, surgical devices, and orthopedics.
        Enables instant out-of-the-box offline testing without downloading bulk FDA archives.
        """
        # 1. Establishments
        establishments = [
            ("1820334", "Ethicon Endo-Surgery, LLC", "1820334", "1002345", "475 West Street", "Cincinnati", "OH", "45242", "US"),
            ("2183427", "Medtronic Vascular", "2183427", "1005678", "3576 Unocal Place", "Santa Rosa", "CA", "95403", "US"),
            ("1002019", "Boston Scientific Corporation", "1002019", "1009988", "One Scimed Place", "Maple Grove", "MN", "55311", "US"),
            ("3003928", "Stryker Instruments", "3003928", "1001122", "1941 Stryker Way", "Portage", "MI", "49002", "US"),
            ("1419308", "Abbott Laboratories / Vascular", "1419308", "1004455", "3200 Lakeside Drive", "Santa Clara", "CA", "95054", "US"),
        ]
        conn = self._get_connection()
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO ref_establishment
                (registration_number, firm_name, fei_number, owner_operator_number, address, city, state, zip_code, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, establishments)

            # 2. Premarket Submissions (510k & PMA)
            pmarket = [
                ("K201452", "510(k)", "Ethicon Endo-Surgery, LLC", "ECHELON FLEX Powered Plus Endopath Stapler", "GAG", "2020-07-15"),
                ("K181234", "510(k)", "Ethicon Endo-Surgery, LLC", "ENFOLD Reloadable Laparoscopic Stapler", "GAG", "2018-09-20"),
                ("P160002", "PMA", "Medtronic Vascular", "Resolute Onyx Zotarolimus-Eluting Coronary Stent System", "NIQ", "2017-04-27"),
                ("P980012", "PMA", "Boston Scientific Corporation", "SYNERGY Everolimus-Eluting Platinum Chromium Coronary Stent", "NIQ", "2015-10-05"),
                ("K192837", "510(k)", "Boston Scientific Corporation", "NC Emerge PTCA Dilatation Catheter", "LIT", "2019-12-03"),
                ("K170119", "510(k)", "Stryker Instruments", "System 8 Precision Power Tool System", "HBE", "2017-06-12"),
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO ref_pmarket
                (premarket_number, premarket_type, applicant_name, device_name, product_code, decision_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, pmarket)

            # 3. GUDID Unique Device Identifiers (UDI-DI)
            gudid = [
                ("00884521034812", "ECHELON FLEX 60", "ECH60A", "Ethicon Endo-Surgery, LLC", "D201452", "GAG", "008845210", "1820334"),
                ("00643169871234", "Resolute Onyx 3.0x18mm", "RONYX3018", "Medtronic Vascular", "D160002", "NIQ", "006431698", "2183427"),
                ("00762145998877", "SYNERGY Monorail 2.50mm", "SYN250", "Boston Scientific Corporation", "D980012", "NIQ", "007621459", "1002019"),
                ("00300392811223", "System 8 Rotary Handpiece", "S8-ROT", "Stryker Instruments", "D170119", "HBE", "003003928", "3003928"),
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO ref_gudid
                (udi_di, brand_name, version_model, company_name, listing_number, product_code, duns_number, fei_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, gudid)

            # 4. Device Listings
            listings = [
                ("D201452", "GAG", "Echelon Flex Endopath", "Ethicon Endo-Surgery, LLC", "K201452"),
                ("D201452", "GAG", "Endocutter 60", "Ethicon Endo-Surgery, LLC", "K201452"),
                ("D160002", "NIQ", "Resolute Onyx Stent", "Medtronic Vascular", "P160002"),
                ("D980012", "NIQ", "Synergy Stent System", "Boston Scientific Corporation", "P980012"),
                ("D192837", "LIT", "NC Emerge Balloon Catheter", "Boston Scientific Corporation", "K192837"),
            ]
            conn.executemany("""
                INSERT INTO ref_listing
                (listing_number, product_code, proprietary_name, firm_name, pmn_number)
                VALUES (?, ?, ?, ?, ?)
            """, listings)

            conn.commit()
        finally:
            self._release_connection(conn)

    def ingest_establishment_file(self, filepath: str, delimiter: str = "|") -> int:
        """
        Ingests official FDA establishment registration file (foirl/establishment.txt).
        Returns count of rows imported.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        conn = self._get_connection()
        count = 0
        try:
            with open(filepath, "r", encoding="latin-1", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = [h.strip().upper() for h in next(reader, [])]

                col_map = {name: idx for idx, name in enumerate(header)}
                reg_idx = col_map.get("REGISTRATION_NUMBER", col_map.get("REG_NO", -1))
                fei_idx = col_map.get("FEI_NUMBER", col_map.get("FEI", -1))
                name_idx = col_map.get("ESTABLISHMENT_NAME", col_map.get("FIRM_NAME", col_map.get("NAME", -1)))
                addr_idx = col_map.get("ADDRESS", col_map.get("ADDRESS_1", -1))
                city_idx = col_map.get("CITY", -1)
                state_idx = col_map.get("STATE_CODE", col_map.get("STATE", -1))
                zip_idx = col_map.get("ZIP_CODE", col_map.get("ZIP", -1))
                country_idx = col_map.get("COUNTRY_CODE", col_map.get("COUNTRY", -1))

                batch = []
                for row in reader:
                    if not row or reg_idx == -1 or reg_idx >= len(row) or not row[reg_idx].strip():
                        continue
                    reg_num = row[reg_idx].strip()
                    firm_name = row[name_idx].strip() if name_idx != -1 and name_idx < len(row) else "Unknown"
                    fei = row[fei_idx].strip() if fei_idx != -1 and fei_idx < len(row) else None
                    addr = row[addr_idx].strip() if addr_idx != -1 and addr_idx < len(row) else None
                    city = row[city_idx].strip() if city_idx != -1 and city_idx < len(row) else None
                    state = row[state_idx].strip() if state_idx != -1 and state_idx < len(row) else None
                    zcode = row[zip_idx].strip() if zip_idx != -1 and zip_idx < len(row) else None
                    country = row[country_idx].strip() if country_idx != -1 and country_idx < len(row) else "US"

                    batch.append((reg_num, fei, None, firm_name, addr, city, state, zcode, country))
                    if len(batch) >= 5000:
                        conn.executemany("""
                            INSERT OR REPLACE INTO ref_establishment
                            (registration_number, fei_number, owner_operator_number, firm_name, address, city, state, zip_code, country)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch)
                        count += len(batch)
                        batch.clear()

                if batch:
                    conn.executemany("""
                        INSERT OR REPLACE INTO ref_establishment
                        (registration_number, fei_number, owner_operator_number, firm_name, address, city, state, zip_code, country)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    count += len(batch)

            conn.commit()
            return count
        finally:
            self._release_connection(conn)

    def ingest_pmn_file(self, filepath: str, delimiter: str = "|") -> int:
        """
        Ingests official FDA 510(k) file (e.g. pmn96cur.txt).
        Returns count of rows imported.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        conn = self._get_connection()
        count = 0
        try:
            with open(filepath, "r", encoding="latin-1", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = [h.strip().upper() for h in next(reader, [])]
                col_map = {name: idx for idx, name in enumerate(header)}

                pmn_idx = col_map.get("KNUMBER", col_map.get("PMN_NUM", -1))
                app_idx = col_map.get("APPLICANT", -1)
                dev_idx = col_map.get("DEVICENAME", col_map.get("DEVICE_NAME", -1))
                pro_idx = col_map.get("PRODUCTCODE", col_map.get("PRODUCT_CODE", -1))
                date_idx = col_map.get("DATECOMPLETED", col_map.get("DATE_DECISION", -1))

                batch = []
                for row in reader:
                    if not row or pmn_idx == -1 or pmn_idx >= len(row) or not row[pmn_idx].strip():
                        continue
                    pmn = row[pmn_idx].strip().upper()
                    app = row[app_idx].strip() if app_idx != -1 and app_idx < len(row) else ""
                    dev = row[dev_idx].strip() if dev_idx != -1 and dev_idx < len(row) else ""
                    pro = row[pro_idx].strip().upper() if pro_idx != -1 and pro_idx < len(row) else None
                    dec = row[date_idx].strip() if date_idx != -1 and date_idx < len(row) else None

                    batch.append((pmn, "510(k)", app, dev, pro, dec))
                    if len(batch) >= 5000:
                        conn.executemany("""
                            INSERT OR REPLACE INTO ref_pmarket
                            (premarket_number, premarket_type, applicant_name, device_name, product_code, decision_date)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, batch)
                        count += len(batch)
                        batch.clear()

                if batch:
                    conn.executemany("""
                        INSERT OR REPLACE INTO ref_pmarket
                        (premarket_number, premarket_type, applicant_name, device_name, product_code, decision_date)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, batch)
                    count += len(batch)

            conn.commit()
            return count
        finally:
            self._release_connection(conn)

    # =========================================================================
    # CORE WATERFALL RESOLUTION ENGINE
    # =========================================================================

    @staticmethod
    def _clean_udi(raw_udi: Optional[str]) -> Optional[str]:
        if not raw_udi:
            return None
        cleaned = DeviceRegistryLinker.RE_UDI_CLEAN.sub("", raw_udi)
        return cleaned if len(cleaned) >= 6 else None

    @staticmethod
    def _extract_premarket_key(raw_val: Optional[str]) -> Optional[str]:
        if not raw_val:
            return None
        m = DeviceRegistryLinker.RE_PREMARKET.search(raw_val.strip())
        return m.group(1).upper() if m else None

    @staticmethod
    def _extract_report_number_reg_no(report_number: Optional[str]) -> Optional[str]:
        if not report_number:
            return None
        m = DeviceRegistryLinker.RE_MFR_REPORT_NUM.match(report_number.strip())
        if m:
            prefix = m.group(1)
            # MedWatch voluntary prefix MW... does not qualify
            if not prefix.upper().startswith("MW"):
                return prefix
        return None

    @staticmethod
    def _token_jaccard_similarity(str1: str, str2: str) -> float:
        """Computes Jaccard word token similarity after stripping medical punctuation."""
        def tokenize(s: str) -> Set[str]:
            tokens = re.findall(r"[A-Za-z0-9]+", s.lower())
            # Filter non-informative stop terms
            stops = {"the", "and", "or", "of", "system", "device", "model", "inc", "corp", "llc", "co", "ltd"}
            return {t for t in tokens if t not in stops and len(t) > 1}

        t1, t2 = tokenize(str1), tokenize(str2)
        if not t1 or not t2:
            return 0.0
        intersection = t1.intersection(t2)
        union = t1.union(t2)
        return len(intersection) / len(union)

    def resolve_report(
        self,
        mdr_report_key: str,
        report_number: Optional[str] = None,
        pma_pmn_num: Optional[str] = None,
        udi_di: Optional[str] = None,
        brand_name: Optional[str] = None,
        model_number: Optional[str] = None,
        product_code: Optional[str] = None,
        manufacturer_name: Optional[str] = None,
    ) -> DeviceResolutionResult:
        """
        Executes the 4-tier waterfall resolution for a single report.

        Waterfall Order:
        1. Tier 1 (UDI-DI -> AccessGUDID)
        2. Tier 2 (PMA/PMN Number -> 510(k)/PMA Registry)
        3. Tier 3 (REPORT_NUMBER -> FDA Establishment Registration Number)
        4. Tier 4 (Product Code + Normalized Brand Heuristic)
        """
        input_keys = {
            "report_number": report_number,
            "pma_pmn_num": pma_pmn_num,
            "udi_di": udi_di,
            "brand_name": brand_name,
            "model_number": model_number,
            "product_code": product_code,
            "manufacturer_name": manufacturer_name,
        }

        conn = self._get_connection()
        try:
            # -----------------------------------------------------------------
            # TIER 1: UDI-DI (AccessGUDID 1-to-1 Barcode Match)
            # -----------------------------------------------------------------
            clean_udi = self._clean_udi(udi_di)
            if clean_udi:
                row = conn.execute("""
                    SELECT brand_name, version_model, company_name, listing_number,
                           product_code, duns_number, fei_number
                    FROM ref_gudid WHERE udi_di = ? LIMIT 1
                """, (clean_udi,)).fetchone()

                if row:
                    dev = RegisteredDevice(
                        udi_di=clean_udi,
                        listing_number=row[3],
                        proprietary_name=row[0],
                        product_code=row[4] or product_code,
                        applicant_or_labeler=row[2],
                    )
                    mfr = RegisteredManufacturer(
                        fei_number=row[6],
                        name=row[2],
                        source_table="gudid",
                    )
                    return DeviceResolutionResult(
                        mdr_report_key=str(mdr_report_key),
                        match_tier=MatchTier.TIER_1_UDI,
                        is_affirmative=True,
                        confidence_score=1.0,
                        manufacturer=mfr,
                        device=dev,
                        input_keys=input_keys,
                        notes=f"Deterministic AccessGUDID match for UDI-DI: {clean_udi}",
                    )

            # -----------------------------------------------------------------
            # TIER 2: Premarket Clearance (510(k), PMA, De Novo)
            # -----------------------------------------------------------------
            clean_pmn = self._extract_premarket_key(pma_pmn_num)
            if clean_pmn:
                row = conn.execute("""
                    SELECT premarket_type, applicant_name, device_name, product_code, decision_date
                    FROM ref_pmarket WHERE premarket_number = ? LIMIT 1
                """, (clean_pmn,)).fetchone()

                if row:
                    dev = RegisteredDevice(
                        premarket_number=clean_pmn,
                        premarket_type=row[0],
                        applicant_or_labeler=row[1],
                        proprietary_name=row[2],
                        product_code=row[3] or product_code,
                        decision_date=row[4],
                    )
                    # Attempt to enrich manufacturer with establishment details if name matches
                    est_row = conn.execute("""
                        SELECT registration_number, fei_number, firm_name, address, city, state, country
                        FROM ref_establishment WHERE firm_name = ? LIMIT 1
                    """, (row[1],)).fetchone()

                    if est_row:
                        mfr = RegisteredManufacturer(
                            registration_number=est_row[0],
                            fei_number=est_row[1],
                            name=est_row[2],
                            address=est_row[3],
                            city=est_row[4],
                            state=est_row[5],
                            country=est_row[6],
                            source_table="establishment",
                        )
                    else:
                        mfr = RegisteredManufacturer(
                            name=row[1],
                            source_table="pmarket",
                        )

                    return DeviceResolutionResult(
                        mdr_report_key=str(mdr_report_key),
                        match_tier=MatchTier.TIER_2_PREMARKET,
                        is_affirmative=True,
                        confidence_score=1.0,
                        manufacturer=mfr,
                        device=dev,
                        input_keys=input_keys,
                        notes=f"Deterministic premarket clearance match for {row[0]}: {clean_pmn} ({row[1]})",
                    )

            # -----------------------------------------------------------------
            # TIER 3: Mandatory Manufacturer Report Number (21 CFR § 803.52)
            # -----------------------------------------------------------------
            reg_num = self._extract_report_number_reg_no(report_number)
            if reg_num:
                row = conn.execute("""
                    SELECT registration_number, fei_number, owner_operator_number, firm_name,
                           address, city, state, zip_code, country
                    FROM ref_establishment
                    WHERE registration_number = ? OR fei_number = ?
                    LIMIT 1
                """, (reg_num, reg_num)).fetchone()

                if row:
                    mfr = RegisteredManufacturer(
                        registration_number=row[0],
                        fei_number=row[1],
                        owner_operator_number=row[2],
                        name=row[3],
                        address=row[4],
                        city=row[5],
                        state=row[6],
                        zip_code=row[7],
                        country=row[8],
                        source_table="establishment",
                    )
                    dev = RegisteredDevice(
                        proprietary_name=brand_name,
                        product_code=product_code,
                        applicant_or_labeler=row[3],
                    ) if brand_name else None

                    return DeviceResolutionResult(
                        mdr_report_key=str(mdr_report_key),
                        match_tier=MatchTier.TIER_3_REPORT_NUMBER,
                        is_affirmative=True,
                        confidence_score=1.0,
                        manufacturer=mfr,
                        device=dev,
                        input_keys=input_keys,
                        notes=f"Deterministic establishment registration {reg_num} extracted from REPORT_NUMBER: {report_number}",
                    )

            # -----------------------------------------------------------------
            # TIER 4: Product Code + Fuzzy Brand Name Heuristic
            # -----------------------------------------------------------------
            if product_code and brand_name:
                p_code = product_code.strip().upper()
                candidates = conn.execute("""
                    SELECT listing_number, proprietary_name, firm_name, pmn_number
                    FROM ref_listing WHERE product_code = ?
                """, (p_code,)).fetchall()

                best_match = None
                best_sim = 0.0

                for c in candidates:
                    sim = self._token_jaccard_similarity(brand_name, c[1])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = c

                # Also inspect premarket candidates
                pm_candidates = conn.execute("""
                    SELECT premarket_number, device_name, applicant_name
                    FROM ref_pmarket WHERE product_code = ?
                """, (p_code,)).fetchall()

                best_pm = None
                for pm in pm_candidates:
                    sim = self._token_jaccard_similarity(brand_name, pm[1])
                    if sim > best_sim:
                        best_sim = sim
                        best_pm = pm
                        best_match = None

                if best_match and best_sim >= 0.40:
                    dev = RegisteredDevice(
                        listing_number=best_match[0],
                        proprietary_name=best_match[1],
                        product_code=p_code,
                        applicant_or_labeler=best_match[2],
                        premarket_number=best_match[3],
                    )
                    mfr = RegisteredManufacturer(
                        name=best_match[2],
                        source_table="listing",
                    ) if best_match[2] else None

                    return DeviceResolutionResult(
                        mdr_report_key=str(mdr_report_key),
                        match_tier=MatchTier.TIER_4_HEURISTIC,
                        is_affirmative=False,
                        confidence_score=round(best_sim, 3),
                        manufacturer=mfr,
                        device=dev,
                        input_keys=input_keys,
                        notes=f"Probabilistic heuristic match on procode '{p_code}' (token Jaccard: {best_sim:.2f}) with listing '{best_match[1]}'",
                    )
                elif best_pm and best_sim >= 0.40:
                    dev = RegisteredDevice(
                        premarket_number=best_pm[0],
                        proprietary_name=best_pm[1],
                        product_code=p_code,
                        applicant_or_labeler=best_pm[2],
                    )
                    mfr = RegisteredManufacturer(
                        name=best_pm[2],
                        source_table="pmarket",
                    )
                    return DeviceResolutionResult(
                        mdr_report_key=str(mdr_report_key),
                        match_tier=MatchTier.TIER_4_HEURISTIC,
                        is_affirmative=False,
                        confidence_score=round(best_sim, 3),
                        manufacturer=mfr,
                        device=dev,
                        input_keys=input_keys,
                        notes=f"Probabilistic heuristic match on procode '{p_code}' (token Jaccard: {best_sim:.2f}) with premarket clearance '{best_pm[1]}'",
                    )

            # -----------------------------------------------------------------
            # UNRESOLVED
            # -----------------------------------------------------------------
            return DeviceResolutionResult(
                mdr_report_key=str(mdr_report_key),
                match_tier=MatchTier.UNRESOLVED,
                is_affirmative=False,
                confidence_score=0.0,
                manufacturer=None,
                device=None,
                input_keys=input_keys,
                notes="No affirmative regulatory key found and heuristic similarity score was below threshold.",
            )

        finally:
            self._release_connection(conn)

    def resolve_from_metadata(self, metadata: Dict[str, Any]) -> DeviceResolutionResult:
        """Resolves report using dictionary returned by MDRDatabase.get_metadata()."""
        return self.resolve_report(
            mdr_report_key=metadata.get("mdr_report_key", "UNKNOWN"),
            report_number=metadata.get("report_number"),
            pma_pmn_num=metadata.get("pma_pmn_num"),
            udi_di=metadata.get("udi_di"),
            brand_name=metadata.get("brand_name"),
            model_number=metadata.get("model_number"),
            product_code=metadata.get("product_code"),
            manufacturer_name=metadata.get("manufacturer_name"),
        )

    def resolve_batch(self, items: Sequence[Union[Dict[str, Any], Any]]) -> List[DeviceResolutionResult]:
        """Resolves a list of metadata dictionaries or MAUDE input records."""
        results = []
        for item in items:
            if isinstance(item, dict):
                results.append(self.resolve_from_metadata(item))
            elif hasattr(item, "mdr_report_key"):
                results.append(self.resolve_report(
                    mdr_report_key=getattr(item, "mdr_report_key", "UNKNOWN"),
                    brand_name=getattr(item, "brand_name", None),
                    product_code=getattr(item, "product_code", None),
                ))
        return results

    # =========================================================================
    # MDRDATABASE BATCH LINKING INTEGRATION
    # =========================================================================

    def link_mdr_database(
        self,
        mdr_db: Any,
        table_name: str = "report_registry_links",
        limit: Optional[int] = None,
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Scans an existing MDRDatabase, executes waterfall resolution for reports,
        and saves affirmative and heuristic associations into a dedicated table.
        Does not modify the original raw tables.
        """
        conn = mdr_db._get_connection()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    mdr_report_key TEXT PRIMARY KEY,
                    match_tier TEXT NOT NULL,
                    is_affirmative INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    registration_number TEXT,
                    manufacturer_name TEXT,
                    premarket_number TEXT,
                    udi_di TEXT,
                    resolved_brand_name TEXT,
                    notes TEXT,
                    linked_at TEXT NOT NULL
                )
            """)
            conn.commit()

            # Query candidate keys with joined device & master attributes
            limit_sql = f"LIMIT {limit}" if limit else ""
            cursor = conn.execute(f"""
                SELECT m.mdr_report_key, m.report_number, d.pma_pmn_num, d.udi_di,
                       d.brand_name, d.model_number, d.product_code, d.manufacturer_name
                FROM mdr_master m
                LEFT JOIN mdr_device d ON m.mdr_report_key = d.mdr_report_key
                {limit_sql}
            """)

            rows = cursor.fetchall()
            total_records = len(rows)

            tier_counts: Dict[str, int] = {t.value: 0 for t in MatchTier}
            saved_batch = []
            now_iso = datetime.now(timezone.utc).isoformat()

            for idx, r in enumerate(rows, 1):
                key, rep_num, pmn, udi, brand, model, procode, mfr = r
                res = self.resolve_report(
                    mdr_report_key=key,
                    report_number=rep_num,
                    pma_pmn_num=pmn,
                    udi_di=udi,
                    brand_name=brand,
                    model_number=model,
                    product_code=procode,
                    manufacturer_name=mfr,
                )
                tier_counts[res.match_tier.value] += 1

                reg_no = res.manufacturer.registration_number if res.manufacturer else None
                mfr_name = res.manufacturer.name if res.manufacturer else None
                p_num = res.device.premarket_number if res.device else None
                u_di = res.device.udi_di if res.device else None
                b_name = res.device.proprietary_name if res.device else None

                saved_batch.append((
                    res.mdr_report_key,
                    res.match_tier.value,
                    1 if res.is_affirmative else 0,
                    res.confidence_score,
                    reg_no,
                    mfr_name,
                    p_num,
                    u_di,
                    b_name,
                    res.notes,
                    now_iso,
                ))

                if len(saved_batch) >= batch_size:
                    conn.executemany(f"""
                        INSERT OR REPLACE INTO {table_name}
                        (mdr_report_key, match_tier, is_affirmative, confidence_score,
                         registration_number, manufacturer_name, premarket_number,
                         udi_di, resolved_brand_name, notes, linked_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, saved_batch)
                    conn.commit()
                    saved_batch.clear()

            if saved_batch:
                conn.executemany(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (mdr_report_key, match_tier, is_affirmative, confidence_score,
                     registration_number, manufacturer_name, premarket_number,
                     udi_di, resolved_brand_name, notes, linked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, saved_batch)
                conn.commit()

            affirmative_count = (
                tier_counts[MatchTier.TIER_1_UDI.value] +
                tier_counts[MatchTier.TIER_2_PREMARKET.value] +
                tier_counts[MatchTier.TIER_3_REPORT_NUMBER.value]
            )

            stats = {
                "total_records_analyzed": total_records,
                "affirmative_matches": affirmative_count,
                "heuristic_matches": tier_counts[MatchTier.TIER_4_HEURISTIC.value],
                "unresolved": tier_counts[MatchTier.UNRESOLVED.value],
                "tier_breakdown": tier_counts,
                "destination_table": table_name,
            }

            if verbose:
                print("=" * 70)
                print("SPOTLIGHT DEVICE & MANUFACTURER REGISTRY LINK SUMMARY")
                print("=" * 70)
                print(f"Total Reports Analyzed: {total_records:,}")
                print(f"Affirmative Matches:    {affirmative_count:,} ({(affirmative_count / max(total_records, 1)) * 100:.1f}%)")
                print(f"  - Tier 1 (UDI-DI):    {tier_counts[MatchTier.TIER_1_UDI.value]:,}")
                print(f"  - Tier 2 (Premarket): {tier_counts[MatchTier.TIER_2_PREMARKET.value]:,}")
                print(f"  - Tier 3 (Report No): {tier_counts[MatchTier.TIER_3_REPORT_NUMBER.value]:,}")
                print(f"Heuristic Matches:      {tier_counts[MatchTier.TIER_4_HEURISTIC.value]:,}")
                print(f"Unresolved:             {tier_counts[MatchTier.UNRESOLVED.value]:,}")
                print("=" * 70)

            return stats
        finally:
            conn.close()

    def stats(self) -> Dict[str, int]:
        """Returns row counts across reference tables."""
        conn = self._get_connection()
        try:
            return {
                "establishments": conn.execute("SELECT COUNT(*) FROM ref_establishment").fetchone()[0],
                "premarket_submissions": conn.execute("SELECT COUNT(*) FROM ref_pmarket").fetchone()[0],
                "gudid_records": conn.execute("SELECT COUNT(*) FROM ref_gudid").fetchone()[0],
                "device_listings": conn.execute("SELECT COUNT(*) FROM ref_listing").fetchone()[0],
            }
        finally:
            self._release_connection(conn)
