"""
Unit tests for ontology normalization and IMDRF term matching.
"""

from spotlight.ontology import OntologyMatcher
from spotlight.schemas import CategoryType


def test_ontology_operational_problem_matching():
    matcher = OntologyMatcher()
    span = "the balloon burst under pressure"
    matches = matcher.match_span(span, category=CategoryType.OPERATIONAL_PROBLEM, top_k=2)

    assert len(matches) > 0
    top = matches[0]
    assert top.code == "A0414"
    assert "Material Split, Cut or Torn" in top.preferred_term
    assert top.ontology == "IMDRF_Annex_A"
    assert top.similarity_score >= 0.8


def test_ontology_manufacturing_issue_matching():
    matcher = OntologyMatcher()
    span = "particulate in package observed prior to use"
    matches = matcher.match_span(span, category=CategoryType.MANUFACTURING_ISSUE, top_k=2)

    assert len(matches) > 0
    top = matches[0]
    assert top.code == "C1901"
    assert "Foreign Material / Particulate" in top.preferred_term
    assert top.ontology == "IMDRF_Annex_C"


def test_ontology_adverse_event_matching():
    matcher = OntologyMatcher()
    span = "patient suffered severe arterial dissection"
    matches = matcher.match_span(span, category=CategoryType.ADVERSE_EVENT, top_k=2)

    assert len(matches) > 0
    top = matches[0]
    assert top.code == "E2401"
    assert "Vascular Dissection" in top.preferred_term
    assert top.ontology == "IMDRF_Annex_E"


def test_ontology_unmatched_returns_empty():
    matcher = OntologyMatcher()
    span = "routine meeting held with clinical sales rep"
    matches = matcher.match_span(span, category=CategoryType.OPERATIONAL_PROBLEM, threshold=0.7)
    assert len(matches) == 0


def test_ontology_clinical_intervention_matching():
    matcher = OntologyMatcher()
    span = "patient required emergent exploratory laparotomy"
    matches = matcher.match_span(span, category=CategoryType.CLINICAL_INTERVENTIONS, top_k=2)

    assert len(matches) > 0
    top = matches[0]
    assert top.code == "F19"
    assert "Surgical Intervention" in top.preferred_term
    assert top.ontology == "IMDRF_Annex_F"


def test_ontology_dark_particulate_matching():
    matcher = OntologyMatcher()
    span = "visible dark particulate matter"
    matches = matcher.match_span(span, category=CategoryType.MANUFACTURING_ISSUE, top_k=2)

    assert len(matches) > 0
    top = matches[0]
    assert top.code == "C1901"
    assert "Foreign Material / Particulate" in top.preferred_term
    assert top.ontology == "IMDRF_Annex_C"


def test_ontology_failure_to_cut_and_fire_codes():
    matcher = OntologyMatcher()

    # Test failure to cut -> A050702
    cut_matches = matcher.match_span("failure to cut", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(cut_matches) > 0
    assert cut_matches[0].code == "A050702"
    assert "Failure to Cut" in cut_matches[0].preferred_term

    # Test failure to fire -> A050501
    fire_matches = matcher.match_span("failure to fire", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(fire_matches) > 0
    assert fire_matches[0].code == "A050501"
    assert "Failure to Fire" in fire_matches[0].preferred_term

    # Test misfired -> A050502
    misfire_matches = matcher.match_span("misfired", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(misfire_matches) > 0
    assert misfire_matches[0].code == "A050502"
    assert "Misfire" in misfire_matches[0].preferred_term


def test_ontology_fracture_vs_crack_vs_burst():
    matcher = OntologyMatcher()

    # Fracture -> A040101
    frac = matcher.match_span("collar broke", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(frac) > 0
    assert frac[0].code == "A040101"
    assert "Fracture" in frac[0].preferred_term

    # Crack -> A0404
    crack = matcher.match_span("hub cracked", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(crack) > 0
    assert crack[0].code == "A0404"
    assert "Crack" in crack[0].preferred_term

    # Balloon burst -> A0414
    burst = matcher.match_span("balloon burst", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(burst) > 0
    assert burst[0].code == "A0414"
    assert "Material Split, Cut or Torn" in burst[0].preferred_term


def test_ontology_detachment_vs_dislodged_vs_deployment():
    matcher = OntologyMatcher()

    # Detachment in vivo -> A0501
    det = matcher.match_span("detached into", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(det) > 0
    assert det[0].code == "A0501"
    assert "Detachment" in det[0].preferred_term

    # Stent dislodgement -> A051201
    dislodged = matcher.match_span("stent dislodged", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(dislodged) > 0
    assert dislodged[0].code == "A051201"
    assert "Device Dislodged or Dislocated" in dislodged[0].preferred_term

    # Deployment failure -> A150101
    deploy = matcher.match_span("failed to deploy", category=CategoryType.OPERATIONAL_PROBLEM, top_k=1)
    assert len(deploy) > 0
    assert deploy[0].code == "A150101"
    assert "Activation Failure / Failure to Deploy" in deploy[0].preferred_term


def test_ontology_tissue_injury_granularity():
    matcher = OntologyMatcher()

    # Laceration -> E2101
    lac = matcher.match_span("tissue laceration", category=CategoryType.ADVERSE_EVENT, top_k=1)
    assert len(lac) > 0
    assert lac[0].code == "E2101"
    assert "Tissue Laceration" in lac[0].preferred_term

    # Dural tear -> E2103
    tear = matcher.match_span("dural tear", category=CategoryType.ADVERSE_EVENT, top_k=1)
    assert len(tear) > 0
    assert tear[0].code == "E2103"
    assert "Tissue / Dural Tear" in tear[0].preferred_term

    # Vessel perforation -> E2114
    perf = matcher.match_span("vessel perforation", category=CategoryType.ADVERSE_EVENT, top_k=1)
    assert len(perf) > 0
    assert perf[0].code == "E2114"
    assert "Perforation" in perf[0].preferred_term

    # Hemorrhage -> E0506
    hem = matcher.match_span("acute hemorrhage", category=CategoryType.ADVERSE_EVENT, top_k=1)
    assert len(hem) > 0
    assert hem[0].code == "E0506"
    assert "Hemorrhage / Bleeding" in hem[0].preferred_term

    # Hemodynamic instability -> E0401
    hemo = matcher.match_span("hemodynamic instability", category=CategoryType.ADVERSE_EVENT, top_k=1)
    assert len(hemo) > 0
    assert hemo[0].code == "E0401"
    assert "Hypotension / Shock / Hemodynamic Instability" in hemo[0].preferred_term


def test_ontology_intervention_granularity():
    matcher = OntologyMatcher()

    # Fragment retrieval -> F1903
    retrieval = matcher.match_span("snare catheter retrieval", category=CategoryType.CLINICAL_INTERVENTIONS, top_k=1)
    assert len(retrieval) > 0
    assert retrieval[0].code == "F1903"
    assert "Device Explantation / Fragment Retrieval" in retrieval[0].preferred_term

    # Surgical revision / CABG -> F1901
    cabg = matcher.match_span("coronary artery bypass graft", category=CategoryType.CLINICAL_INTERVENTIONS, top_k=1)
    assert len(cabg) > 0
    assert cabg[0].code == "F1901"
    assert "Surgical Revision / Revascularization" in cabg[0].preferred_term

    # Blood product transfusion -> F2302
    trans = matcher.match_span("blood transfusion", category=CategoryType.CLINICAL_INTERVENTIONS, top_k=1)
    assert len(trans) > 0
    assert trans[0].code == "F2302"
    assert "Blood Product Transfusion" in trans[0].preferred_term

    # Prolonged surgery -> F1908
    prol = matcher.match_span("prolonged anesthesia time", category=CategoryType.CLINICAL_INTERVENTIONS, top_k=1)
    assert len(prol) > 0
    assert prol[0].code == "F1908"
    assert "Prolonged Surgery / Extended Procedure" in prol[0].preferred_term

    # Pharmacological support -> F2301
    med = matcher.match_span("vasopressors", category=CategoryType.CLINICAL_INTERVENTIONS, top_k=1)
    assert len(med) > 0
    assert med[0].code == "F2301"
    assert "Pharmacological / Medical Support" in med[0].preferred_term

