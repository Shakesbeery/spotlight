"""
Demonstration runner for Pathway A (5 New Diverse Narratives).
Evaluates extraction of:
1. Operational Problems (IMDRF Annex A)
2. Manufacturing Issues & Atomic Particulate Matter (IMDRF Annex C)
3. Adverse Events (IMDRF Annex E)
4. Clinical Interventions & Causal Linking (IMDRF Annex F)
5. Other
"""

import json
from spotlight.pipeline import MAUDEExtractionPipeline
from spotlight.schemas import MAUDERecordInput


def get_new_evaluation_narratives():
    return [
        # Narrative 1: Cardiovascular Stent Dislodgement -> Snare Retrieval & Emergency CABG
        MAUDERecordInput(
            mdr_report_key="MDR-NEW-001",
            brand_name="Resolute Onyx Zotarolimus-Eluting Stent",
            product_code="NIQ",
            event_type="Injury",
            narrative_text=(
                "THIS REPORT IS BEING SUBMITTED PURSUANT TO 21 CFR PART 803. (b)(4) "
                "EVENT DESCRIPTION: During coronary intervention, while navigating tortuous anatomy, the stent dislodged "
                "from the delivery balloon inside the proximal LAD. The dislodged stent caused an acute coronary dissection "
                "and sudden hemodynamic instability. The interventionalist attempted snare catheter retrieval to capture the "
                "dislodged stent. Due to incomplete capture, the patient underwent emergent coronary artery bypass graft (CABG) "
                "surgery to revascularize the myocardium. "
                "MANUFACTURER EVALUATION: The complaint device was returned. The crimped stent was absent from the balloon."
            ),
        ),

        # Narrative 2: Orthopedic Sterile Prep -> Dark Particulate Matter & Cracked Tray
        MAUDERecordInput(
            mdr_report_key="MDR-NEW-002",
            brand_name="Attune Primary Femoral Component",
            product_code="JWH",
            event_type="Malfunction",
            narrative_text=(
                "THE SUBMISSION OF THIS REPORT DOES NOT CONSTITUTE AN ADMISSION OF PRODUCT DEFECT. "
                "EVENT DESCRIPTION: Prior to surgery, during sterile setup in the OR, the scrub technician identified "
                "visible dark particulate matter adhering to the articular surface of the femoral implant. A crack in the "
                "inner sterile barrier tray was also identified upon secondary inspection. The implant was immediately quarantined "
                "and rejected prior to patient contact. A backup implant was retrieved and opened. No patient was involved "
                "and no patient injury occurred. "
                "MANUFACTURER EVALUATION: Device received for investigation. Microscopic examination confirmed dark particulate matter."
            ),
        ),

        # Narrative 3: Laparoscopic Surgical Stapler Misfire -> Laparotomy & Transfusion
        MAUDERecordInput(
            mdr_report_key="MDR-NEW-003",
            brand_name="Echelon Flex Powered Plus Articulating Stapler",
            product_code="GAG",
            event_type="Injury",
            narrative_text=(
                "EVENT DESCRIPTION: Intraoperatively, during a sigmoid colectomy, the surgical stapler misfired across the mesenteric "
                "vessels and demonstrated a complete failure to cut. The jammed jaw caused severe tissue laceration and acute hemorrhage. "
                "The surgical team performed an emergent exploratory laparotomy to achieve hemostasis. The patient received a blood "
                "transfusion of 2 units of PRBC and required prolonged anesthesia time. "
                "MANUFACTURER EVALUATION: Log analysis of the powered handle indicates a motor drive stall under tissue load."
            ),
        ),

        # Narrative 4: Vascular Access Sheath Hub Fracture -> Fluid Resuscitation & Extended Fluoroscopy
        MAUDERecordInput(
            mdr_report_key="MDR-NEW-004",
            brand_name="DrySeal Select Hydrophilic Introducer Sheath",
            product_code="DYB",
            event_type="Injury",
            narrative_text=(
                "EVENT DESCRIPTION: During endovascular aneurysm repair (EVAR), while advancing the stent-graft main body, the "
                "introducer sheath hub was fractured at the hemostasis valve junction. The hub cracked and leaked large volumes of "
                "saline and blood, leading to sudden hemodynamic instability. The clinical team administered rapid fluid resuscitation "
                "and vasopressors. The delivery was successfully completed with a replacement sheath, but the patient experienced "
                "prolonged operating time and extended anesthesia. "
                "MANUFACTURER EVALUATION: Retained sample showed crack propagating from sidearm sonic weld."
            ),
        ),

        # Narrative 5: Spine Surgery Tulip Head Breakage -> Fragment Retrieval & Dura Repair
        MAUDERecordInput(
            mdr_report_key="MDR-NEW-005",
            brand_name="Viper Prime Fenestrated Pedicle Screw",
            product_code="NKB",
            event_type="Injury",
            narrative_text=(
                "EVENT DESCRIPTION: Intraoperatively, during lumbar arthrodesis, while torquing the set screw, the polyaxial tulip "
                "collar broke off the shank. One fractured piece detached into the epidural space. Upon separation, the sharp metal edge "
                "caused an unintended dural tear. The spine surgeon conducted delicate retrieval of the detached fragment using magnetic "
                "forceps, followed by direct surgical repair of the dura. The operation required prolonged anesthesia time. "
                "MANUFACTURER EVALUATION: Tensile shear analysis confirmed over-torque fracture pattern on the tulip head."
            ),
        ),
    ]


def run():
    pipeline = MAUDEExtractionPipeline(confidence_threshold=0.80, use_mock_slm=True)
    narratives = get_new_evaluation_narratives()

    outputs = []
    for item in narratives:
        out = pipeline.process_record(item)
        outputs.append((item, out))

    return outputs


if __name__ == "__main__":
    results = run()
    for record, out in results:
        print("\n" + "=" * 80)
        print(f"EVALUATION RECORD : {record.mdr_report_key} | {record.brand_name} ({record.product_code})")
        print(f"FDA EVENT TYPE    : {record.event_type}")
        print(f"PIPELINE ROUTE    : {out.execution_path.value} | LATENCY: {out.processing_time_ms:.2f} ms")
        print("=" * 80)

        print("\n[RAW NARRATIVE]:")
        print(record.narrative_text)

        print("\n[CLEANED NARRATIVE (Boilerplate Stripped)]:")
        print(out.cleaned_narrative)

        print("\n[STRUCTURED FINDINGS - 5 CATEGORIES]:")

        print(f"\n1. Operational Problems ({len(out.operational_problems)}):")
        for f in out.operational_problems:
            codes = ", ".join([f"{t.code}: {t.preferred_term} ({t.ontology})" for t in f.normalized_terms]) or "No code"
            comp_str = f" | Component: \"{f.affected_component}\"" if f.affected_component else ""
            alias_str = f" | Aliases: {f.aliases}" if f.aliases else ""
            print(f"   [{f.id}] Verbatim: \"{f.verbatim_span}\"{comp_str}{alias_str} | Timing: {f.temporal_timing.value}")
            print(f"        Ontology: [{codes}]")

        print(f"\n2. Manufacturing Issues & Atomic Particulate ({len(out.manufacturing_issues)}):")
        for f in out.manufacturing_issues:
            codes = ", ".join([f"{t.code}: {t.preferred_term} ({t.ontology})" for t in f.normalized_terms]) or "No code"
            comp_str = f" | Component: \"{f.affected_component}\"" if f.affected_component else ""
            alias_str = f" | Aliases: {f.aliases}" if f.aliases else ""
            print(f"   [{f.id}] Verbatim: \"{f.verbatim_span}\"{comp_str}{alias_str} | Timing: {f.temporal_timing.value}")
            print(f"        Ontology: [{codes}]")

        print(f"\n3. Patient Adverse Events ({len(out.adverse_events)}):")
        for f in out.adverse_events:
            codes = ", ".join([f"{t.code}: {t.preferred_term} ({t.ontology})" for t in f.normalized_terms]) or "No code"
            alias_str = f" | Aliases: {f.aliases}" if f.aliases else ""
            print(f"   [{f.id}] Verbatim: \"{f.verbatim_span}\"{alias_str} | Timing: {f.temporal_timing.value}")
            print(f"        Ontology: [{codes}]")

        print(f"\n4. Clinical / Surgical Interventions ({len(out.clinical_interventions)}):")
        for f in out.clinical_interventions:
            codes = ", ".join([f"{t.code}: {t.preferred_term} ({t.ontology})" for t in f.normalized_terms]) or "No code"
            alias_str = f" | Aliases/Acronyms: {f.aliases}" if f.aliases else ""
            trig = f" -> Triggered by: {f.triggered_by}" if f.triggered_by else ""
            print(f"   [{f.id}] Verbatim: \"{f.verbatim_span}\"{alias_str} | Timing: {f.temporal_timing.value}{trig}")
            print(f"        Ontology: [{codes}]")

        print(f"\n5. Other ({len(out.other)}):")
        for f in out.other:
            print(f"   [{f.id}] Verbatim: \"{f.verbatim_span}\"")
        print("-" * 80)
