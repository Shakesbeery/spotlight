"""
Unit tests for Component Linker and Entity Resolver.
"""

from spotlight.schemas import ExtractedFinding, CategoryType, TemporalTiming, GroundedOntologyTerm
from spotlight.component_linker import ComponentLinker, EntityResolver


def test_component_extraction_and_linkage():
    linker = ComponentLinker()
    narrative = (
        "During surgery, the introducer sheath hub was fractured at the junction. "
        "The polyaxial tulip collar broke off the screw."
    )

    findings = [
        ExtractedFinding(
            id="OP-1",
            category=CategoryType.OPERATIONAL_PROBLEM,
            verbatim_span="hub was fractured",
            start_char=27,
            end_char=44,
            temporal_timing=TemporalTiming.INTRA_USE,
            confidence=0.90,
        ),
        ExtractedFinding(
            id="OP-2",
            category=CategoryType.OPERATIONAL_PROBLEM,
            verbatim_span="collar broke",
            start_char=89,
            end_char=101,
            temporal_timing=TemporalTiming.INTRA_USE,
            confidence=0.90,
        ),
    ]

    linked = linker.link_components_to_findings(narrative, findings)
    assert len(linked) == 2
    assert "sheath hub" in linked[0].affected_component.lower()
    assert "tulip collar" in linked[1].affected_component.lower()


def test_reconcile_acronym_true_duplicate():
    narrative = "Patient underwent emergent coronary artery bypass graft (CABG) surgery."

    term = GroundedOntologyTerm(
        code="F0101", preferred_term="Surgical Intervention", ontology="IMDRF_Annex_F", similarity_score=1.0
    )

    findings = [
        ExtractedFinding(
            id="INT-1",
            category=CategoryType.CLINICAL_INTERVENTIONS,
            verbatim_span="coronary artery bypass graft",
            confidence=0.90,
            normalized_terms=[term],
        ),
        ExtractedFinding(
            id="INT-2",
            category=CategoryType.CLINICAL_INTERVENTIONS,
            verbatim_span="CABG",
            confidence=0.90,
            normalized_terms=[term],
        ),
    ]

    resolved = EntityResolver.reconcile_duplicates(findings, narrative)
    # CABG is an acronym of coronary artery bypass graft -> should be merged into 1 finding with alias
    assert len(resolved) == 1
    assert resolved[0].verbatim_span == "coronary artery bypass graft"
    assert "CABG" in resolved[0].aliases


def test_preserve_distinct_manifestations_under_ontology_collapse():
    narrative = "The stapler misfired across the vessel and demonstrated a complete failure to cut."

    term_a0502 = GroundedOntologyTerm(
        code="A0502", preferred_term="Failure to Fire / Jam", ontology="IMDRF_Annex_A", similarity_score=1.0
    )

    findings = [
        ExtractedFinding(
            id="OP-1",
            category=CategoryType.OPERATIONAL_PROBLEM,
            verbatim_span="misfired",
            confidence=0.90,
            affected_component="stapler handle",
            normalized_terms=[term_a0502],
        ),
        ExtractedFinding(
            id="OP-2",
            category=CategoryType.OPERATIONAL_PROBLEM,
            verbatim_span="failure to cut",
            confidence=0.90,
            affected_component="stapler jaw",
            normalized_terms=[term_a0502],
        ),
    ]

    resolved = EntityResolver.reconcile_duplicates(findings, narrative)
    # Both map to A0502, but are distinct physical failure facets -> MUST NOT be merged!
    assert len(resolved) == 2
    assert resolved[0].verbatim_span == "misfired"
    assert resolved[1].verbatim_span == "failure to cut"
