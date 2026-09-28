"""
Ontology normalization and terminology grounding engine.
Maps free-text extracted spans to IMDRF (Annexes A, C, E) and MedDRA standard regulatory codes.
"""

from typing import List, Dict, Tuple, Optional
import difflib
import re
from spotlight.schemas import GroundedOntologyTerm, CategoryType


# Curated foundational registry of official IMDRF and MedDRA terms for medical devices
# IMDRF Annex A: Medical Device Problems (Operational failure modes)
IMDRF_ANNEX_A = [
    {"code": "A0401", "term": "Break", "synonyms": ["break", "broke", "broken", "material break"]},
    {"code": "A040101", "term": "Fracture", "synonyms": ["fracture", "fractured", "broken shaft", "snapped", "collar broke", "hub was fractured", "structural fracture"]},
    {"code": "A040102", "term": "Loss of or Failure to Bond", "synonyms": ["loss of bond", "failure to bond", "weld separation", "adhesive failure", "delamination"]},
    {"code": "A0402", "term": "Burst Container", "synonyms": ["burst container", "fluid reservoir burst", "vessel burst"]},
    {"code": "A0404", "term": "Crack", "synonyms": ["crack", "cracked", "cracked tip", "hub cracked", "crazing", "surface crack", "hairline crack"]},
    {"code": "A0414", "term": "Material Split, Cut or Torn", "synonyms": ["balloon burst", "ruptured", "popped", "bursting", "material split", "torn balloon", "balloon popped", "rupture"]},
    {"code": "A0501", "term": "Detachment of Device or Device Component", "synonyms": ["detached", "separated in vivo", "distal tip fell off", "detached into", "separated into", "component detached"]},
    {"code": "A0504", "term": "Leak / Splash", "synonyms": ["fluid leak", "leaking saline", "extravasation at hub", "leaked", "leakage", "saline leak", "blood leak"]},
    {"code": "A050501", "term": "Failure to Fire", "synonyms": ["failed to fire", "failure to fire", "did not actuate", "failure to discharge"]},
    {"code": "A050502", "term": "Misfire", "synonyms": ["misfired", "misfire"]},
    {"code": "A0506", "term": "Mechanical Jam", "synonyms": ["jammed", "stapler jammed", "mechanical jam", "jam", "jaw jammed"]},
    {"code": "A050702", "term": "Failure to Cut", "synonyms": ["failure to cut", "failed to cut", "unable to cut", "did not cut", "incomplete cut"]},
    {"code": "A051201", "term": "Device Dislodged or Dislocated", "synonyms": ["dislodged", "stent dislodged", "dislocated", "device dislodged", "implant dislodged"]},
    {"code": "A0901", "term": "Occlusion / Clog", "synonyms": ["occluded", "clogged", "lumen blocked", "thrombosed catheter"]},
    {"code": "A1301", "term": "Overheating / High Temperature", "synonyms": ["overheated", "temperature spike", "thermal cutoff", "excessive heat"]},
    {"code": "A1401", "term": "Software / Firmware Error", "synonyms": ["system crash", "error code 502", "frozen screen", "rebooted", "firmware lockup"]},
    {"code": "A150101", "term": "Activation Failure / Failure to Deploy", "synonyms": ["failed to deploy", "did not deploy", "deployment failure", "unable to deploy", "failure to deploy", "activation failure"]},
    {"code": "A150102", "term": "Difficult or Delayed Activation / Deployment", "synonyms": ["delayed deployment", "difficult deployment", "incomplete deployment"]},
]

# IMDRF Annex C: Pre-Use, Manufacturing, Packaging & Investigation Findings
IMDRF_ANNEX_C = [
    {"code": "C0101", "term": "Missing Component", "synonyms": ["component missing", "missing lock collar", "absent cap", "missing part", "missing screw"]},
    {"code": "C0102", "term": "Loose / Disassembled Component", "synonyms": ["loose cap", "loose sealing cap", "sealing cap was loose", "loose screw", "loose collar"]},
    {"code": "C0201", "term": "Packaging Defect", "synonyms": ["damaged packaging", "packaging defect", "damaged box", "torn packaging"]},
    {"code": "C0202", "term": "Seal Integrity / Sterile Barrier Compromised", "synonyms": ["compromised seal", "sterility breach", "damaged sterile barrier", "sterile barrier breach", "torn seal", "seal was torn", "packaging seal was torn", "crack in the inner sterile barrier tray", "crack in inner sterile barrier tray", "compromised sterile barrier"]},
    {"code": "C0203", "term": "Packaging Punctured / Torn", "synonyms": ["torn pouch", "punctured pouch", "punctured packaging", "hole in pouch"]},
    {"code": "C0301", "term": "Material Flaw / Defect", "synonyms": ["defective polymer", "brittle shaft", "manufacturing void"]},
    {"code": "C0401", "term": "Labeling / Packaging Error", "synonyms": ["mislabeled size", "wrong expiry date", "incorrect label", "mislabeled"]},
    {"code": "C1901", "term": "Foreign Material / Particulate", "synonyms": ["particulate in package", "foreign debris", "white speck", "lint on device", "dark particulate", "dark particulate matter", "metallic debris", "foreign particulate", "particulate matter", "visible dark particulate", "black particulate", "metallic shavings", "visible dark particulate matter", "foreign material"]},
    {"code": "C2001", "term": "Dimensional Discrepancy", "synonyms": ["out of tolerance", "incorrect diameter", "oversized lumen"]},
]

# IMDRF Annex E / MedDRA: Clinical Health Effects & Adverse Events
IMDRF_ANNEX_E = [
    {"code": "E0202", "term": "Arrhythmia", "synonyms": ["ventricular fibrillation", "arrhythmia", "tachycardia", "dysrhythmia"]},
    {"code": "E0401", "term": "Hypotension / Shock / Hemodynamic Instability", "synonyms": ["hemodynamic instability", "sudden hemodynamic instability", "hypotension", "circulatory shock", "severe hypotension"]},
    {"code": "E0506", "term": "Hemorrhage / Bleeding", "synonyms": ["severe bleeding", "hemorrhage", "excessive blood loss", "acute hemorrhage", "active bleeding"]},
    {"code": "E0507", "term": "Hematoma", "synonyms": ["hematoma", "retroperitoneal hematoma", "wound hematoma"]},
    {"code": "E0601", "term": "Cardiac Arrest", "synonyms": ["asystole", "cardiac arrest", "cardiac standstill"]},
    {"code": "E0604", "term": "Cardiac Perforation", "synonyms": ["cardiac perforation", "myocardial perforation", "atrial perforation", "ventricular perforation"]},
    {"code": "E0801", "term": "Burn / Thermal Injury", "synonyms": ["skin burn", "thermal tissue necrosis", "cautery burn", "thermal burn"]},
    {"code": "E1901", "term": "Infection / Sepsis", "synonyms": ["infection", "septicemia", "sepsis", "bacteremia", "purulent discharge", "systemic infection", "local infection"]},
    {"code": "E2101", "term": "Tissue Laceration", "synonyms": ["tissue laceration", "laceration", "torn tissue edge", "mesenteric laceration"]},
    {"code": "E2102", "term": "Puncture", "synonyms": ["punctured organ", "puncture", "penetrating puncture", "punctured"]},
    {"code": "E2103", "term": "Tissue / Dural Tear", "synonyms": ["dural tear", "tear", "torn tissue", "pleural tear", "capsular tear"]},
    {"code": "E2114", "term": "Perforation", "synonyms": ["vessel perforation", "perforated wall", "bowel perforation", "arterial perforation", "perforation", "perforated"]},
    {"code": "E2301", "term": "Thrombosis / Embolism", "synonyms": ["air embolism", "thrombus formation", "stroke", "ischemic event", "thrombosis"]},
    {"code": "E2401", "term": "Vascular Dissection", "synonyms": ["arterial dissection", "aortic tear", "intimal flap dissection", "coronary artery dissection", "acute coronary dissection", "dissection"]},
]

# IMDRF Annex F: Health Effects - Clinical Impact & Medical / Surgical Interventions
IMDRF_ANNEX_F = [
    {"code": "F0801", "term": "Hospitalization Required", "synonyms": ["hospitalization", "admitted to hospital", "emergency department transfer"]},
    {"code": "F0802", "term": "Prolonged Hospitalization / ICU Admission", "synonyms": ["extended hospital stay", "icu admission", "admitted to icu", "prolonged hospitalization"]},
    {"code": "F19", "term": "Surgical Intervention", "synonyms": ["exploratory laparotomy", "laparotomy", "sternotomy", "thoracotomy", "emergent surgery", "reoperation", "surgical intervention"]},
    {"code": "F1901", "term": "Surgical Revision / Revascularization", "synonyms": ["cabg", "coronary artery bypass", "coronary artery bypass graft", "surgical repair", "surgical cutdown", "direct surgical repair", "dura repair", "revision surgery"]},
    {"code": "F1903", "term": "Device Explantation / Fragment Retrieval", "synonyms": ["device explanted", "retrieval of fragment", "retrieve the detached fragment", "retrieval of the detached fragment", "explant", "removed implant", "snare retrieval", "snare catheter retrieval", "catheter retrieval"]},
    {"code": "F1905", "term": "Rescue / Bailout Percutaneous Stenting", "synonyms": ["covered stent", "stenting", "emergency stenting", "rescue stent", "bail-out stent", "bailout stenting"]},
    {"code": "F1908", "term": "Prolonged Surgery / Extended Procedure", "synonyms": ["prolonged anesthesia", "prolonged anesthesia time", "extended procedure", "extended operating time", "prolonged procedure", "extended anesthesia", "prolonged operating time"]},
    {"code": "F2301", "term": "Pharmacological / Medical Support", "synonyms": ["vasopressors", "fluid resuscitation", "epinephrine", "blood pressure support", "pharmacological support", "medication administered"]},
    {"code": "F2302", "term": "Blood Product Transfusion", "synonyms": ["blood transfusion", "transfusion", "packed red blood cells", "prbc transfusion", "blood products", "2 units of prbc", "blood transfusion of 2 units of prbc"]},
    {"code": "F2501", "term": "Cardiopulmonary Resuscitation / Life Support", "synonyms": ["cpr", "cardiopulmonary resuscitation", "defibrillation", "intubation", "mechanical ventilation"]},
]


class OntologyMatcher:
    """
    Standardizes verbatim clinical/device phrases to regulatory ontology concepts (IMDRF/MedDRA).
    Supports category-scoped ranking, synonym matching, and fuzzy string similarity.
    """

    def __init__(self):
        self.ontologies: Dict[str, List[Dict]] = {
            "IMDRF_Annex_A": IMDRF_ANNEX_A,
            "IMDRF_Annex_C": IMDRF_ANNEX_C,
            "IMDRF_Annex_E": IMDRF_ANNEX_E,
            "IMDRF_Annex_F": IMDRF_ANNEX_F,
        }
        self._exact_index: Dict[str, List[Tuple[str, str, str]]] = {}
        for ont_name, records in self.ontologies.items():
            for item in records:
                code = item["code"]
                term = item["term"]
                synonyms = item.get("synonyms", [])
                for phrase in [term] + synonyms:
                    k = phrase.lower().strip()
                    if k not in self._exact_index:
                        self._exact_index[k] = []
                    self._exact_index[k].append((code, term, ont_name))

    def _clean_token_set(self, text: str) -> set:
        """Tokenize and remove punctuation/stopwords for similarity calculation."""
        words = re.findall(r"\b\w+\b", text.lower())
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "was", "is", "of"}
        return {w for w in words if w not in stopwords}

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Combines token Jaccard overlap and difflib sequence matching."""
        tokens_a = self._clean_token_set(text_a)
        tokens_b = self._clean_token_set(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        seq_ratio = difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

        # Weighted combination: 60% token overlap, 40% character sequence similarity
        return round(0.6 * jaccard + 0.4 * seq_ratio, 3)

    def match_span(
        self,
        span: str,
        category: Optional[CategoryType] = None,
        top_k: int = 3,
        threshold: float = 0.35,
    ) -> List[GroundedOntologyTerm]:
        """
        Given an extracted free-text span and an optional category, returns the top-k grounded ontology terms.
        """
        candidate_ontologies = []
        if category == CategoryType.OPERATIONAL_PROBLEM:
            candidate_ontologies = ["IMDRF_Annex_A"]
        elif category == CategoryType.MANUFACTURING_ISSUE:
            candidate_ontologies = ["IMDRF_Annex_C"]
        elif category == CategoryType.ADVERSE_EVENT:
            candidate_ontologies = ["IMDRF_Annex_E"]
        elif category == CategoryType.CLINICAL_INTERVENTIONS:
            candidate_ontologies = ["IMDRF_Annex_F"]
        else:
            candidate_ontologies = ["IMDRF_Annex_A", "IMDRF_Annex_C", "IMDRF_Annex_E", "IMDRF_Annex_F"]

        clean_span = span.lower().strip()

        # Fast path 1: Exact match in indexed vocabulary (O(1) lookup)
        if clean_span in self._exact_index:
            exact_matches = []
            seen_exact_codes = set()
            for code, term, ont_name in self._exact_index[clean_span]:
                if ont_name in candidate_ontologies and code not in seen_exact_codes:
                    seen_exact_codes.add(code)
                    exact_matches.append(
                        GroundedOntologyTerm(
                            code=code,
                            preferred_term=term,
                            ontology=ont_name,
                            similarity_score=1.0,
                        )
                    )
            if exact_matches:
                return exact_matches[:top_k]

        matches: List[Tuple[float, GroundedOntologyTerm]] = []

        for ont_name in candidate_ontologies:
            records = self.ontologies[ont_name]
            for item in records:
                code = item["code"]
                term = item["term"]
                synonyms = item.get("synonyms", [])

                # Compare span against primary term
                best_sim = self._compute_similarity(span, term)

                # Compare against all known synonyms
                for syn in synonyms:
                    sim = self._compute_similarity(span, syn)
                    if sim > best_sim:
                        best_sim = sim

                # Substring containment bonus (safe word-boundary guarded)
                clean_term = term.lower()
                for target in [clean_term] + [s.lower() for s in synonyms]:
                    if target in clean_span:
                        best_sim = max(best_sim, 0.85)
                    elif len(clean_span) >= 4 and re.search(rf"\b{re.escape(clean_span)}\b", target):
                        best_sim = max(best_sim, 0.85)

                if best_sim >= threshold:
                    grounded = GroundedOntologyTerm(
                        code=code,
                        preferred_term=term,
                        ontology=ont_name,
                        similarity_score=min(1.0, best_sim),
                    )
                    matches.append((best_sim, grounded))

        # Sort descending by similarity score
        matches.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by code
        seen_codes = set()
        results: List[GroundedOntologyTerm] = []
        for sim, grounded in matches:
            if grounded.code not in seen_codes:
                seen_codes.add(grounded.code)
                results.append(grounded)
            if len(results) >= top_k:
                break

        return results
