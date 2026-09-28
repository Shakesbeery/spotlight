"""
Unit tests for SpotlightProject: Isolated workspaces, cross-project contamination guards,
project-scoped delta pipelines, and Vigipy exports.
"""

import os
import json
import tempfile
import pytest
import spotlight
from spotlight import (
    SpotlightProject,
    ProjectMismatchError,
    ProjectNotFoundError,
    ProjectActiveConflictError,
    MDRDatabase,
    MAUDERecordInput,
)


def test_project_creation_and_manifest(tmp_path):
    projects_dir = tmp_path / "projects"
    raw_db_path = tmp_path / "raw_maude.db"

    # Create project
    proj = SpotlightProject(
        name="stapler_safety_2025",
        base_dir=str(projects_dir),
        raw_db=str(raw_db_path),
        description="Comparative stapler failure modes study",
        tags=["stapler", "surgical"],
    )

    assert proj.name == "stapler_safety_2025"
    assert os.path.exists(proj.project_dir)
    assert os.path.exists(proj.manifest_path)
    assert os.path.exists(proj.extractions_db_path)
    assert os.path.exists(proj.exports_dir)

    manifest = proj.manifest
    assert manifest["name"] == "stapler_safety_2025"
    assert manifest["description"] == "Comparative stapler failure modes study"
    assert "stapler" in manifest["tags"]

    # Test list_projects
    p_list = SpotlightProject.list_projects(base_dir=str(projects_dir))
    assert len(p_list) == 1
    assert p_list[0]["name"] == "stapler_safety_2025"

    # Test open existing
    opened = SpotlightProject.open("stapler_safety_2025", base_dir=str(projects_dir))
    assert opened.name == "stapler_safety_2025"

    # Test open non-existent raises
    with pytest.raises(ProjectNotFoundError):
        SpotlightProject.open("non_existent_project", base_dir=str(projects_dir))

    proj.close()
    opened.close()


def test_project_context_manager_and_active_session(tmp_path):
    projects_dir = tmp_path / "projects"

    proj1 = SpotlightProject("study_alpha", base_dir=str(projects_dir))
    proj2 = SpotlightProject("study_beta", base_dir=str(projects_dir))

    assert SpotlightProject.get_active() is None

    with proj1:
        assert SpotlightProject.get_active() == proj1

        # Attempting to activate a conflicting project in the same context raises error
        with pytest.raises(ProjectActiveConflictError):
            with proj2:
                pass

    # Cleared upon exit
    assert SpotlightProject.get_active() is None

    proj1.close()
    proj2.close()


def test_cross_project_contamination_guard(tmp_path):
    """
    Verifies that outputs generated for Project A cannot accidentally be saved into
    Project B without explicit permission (allow_cross_project=True).
    """
    projects_dir = tmp_path / "projects"
    p_alpha = SpotlightProject("study_alpha", base_dir=str(projects_dir))
    p_beta = SpotlightProject("study_beta", base_dir=str(projects_dir))

    extractor = spotlight.get_default_extractor()

    # Extract finding within alpha
    rec1 = MAUDERecordInput(
        mdr_report_key="REC-001",
        brand_name="AlphaStapler",
        product_code="GAG",
        event_type="Injury",
        narrative_text="Surgical stapler misfired during resection. Acute hemorrhage occurred.",
    )
    output1 = extractor.process_record(rec1)

    # Save to alpha -> succeeds and tags with study_alpha
    p_alpha.save_extractions([output1])
    assert p_alpha.count_extracted() == 1

    loaded_alpha = p_alpha.load_extractions()
    assert len(loaded_alpha) == 1
    assert loaded_alpha[0].project_name == "study_alpha"

    # Attempt to save loaded_alpha outputs into study_beta!
    # Guard check MUST prevent cross-project pollution
    with pytest.raises(ProjectMismatchError) as exc_info:
        p_beta.save_extractions(loaded_alpha)

    assert "Cross-project contamination prevented" in str(exc_info.value)
    assert "study_alpha" in str(exc_info.value)
    assert "study_beta" in str(exc_info.value)

    # Project beta remains clean (0 records)
    assert p_beta.count_extracted() == 0

    # Explicit override with allow_cross_project=True
    p_beta.save_extractions(loaded_alpha, allow_cross_project=True)
    assert p_beta.count_extracted() == 1

    p_alpha.close()
    p_beta.close()


def test_project_scoped_delta_pipeline(tmp_path):
    """
    Verifies that Project A and Project B can both query the same shared raw database,
    while maintaining independent extraction progress and delta anti-joins.
    """
    raw_db_path = tmp_path / "shared_raw.db"
    raw_db = MDRDatabase(db_path=str(raw_db_path))

    # Ingest 3 shared raw records
    dev_txt = tmp_path / "dev.txt"
    dev_txt.write_text("MDR_REPORT_KEY|BRAND_NAME|DEVICE_REPORT_PRODUCT_CODE\nR1|DevA|GAG\nR2|DevB|GAG\nR3|DevC|GAG\n", encoding="latin-1")
    txt_txt = tmp_path / "txt.txt"
    txt_txt.write_text("MDR_REPORT_KEY|TEXT_TYPE_CODE|FOI_TEXT\nR1|D|Misfire.\nR2|D|Jammed.\nR3|D|Bursted.\n", encoding="latin-1")
    mst_txt = tmp_path / "mst.txt"
    mst_txt.write_text("MDR_REPORT_KEY|EVENT_TYPE\nR1|Malfunction\nR2|Injury\nR3|Malfunction\n", encoding="latin-1")

    raw_db.ingest_txt(str(dev_txt), verbose=False)
    raw_db.ingest_txt(str(txt_txt), verbose=False)
    raw_db.ingest_txt(str(mst_txt), verbose=False)
    raw_db.close()

    projects_dir = tmp_path / "projects"
    p_stapler = SpotlightProject("staplers", base_dir=str(projects_dir), raw_db=str(raw_db_path))
    p_balloon = SpotlightProject("balloons", base_dir=str(projects_dir), raw_db=str(raw_db_path))

    extractor = spotlight.get_default_extractor()

    # Step 1: Run p_stapler for first 2 records
    stats1 = p_stapler.orchestrate_pipeline(extractor=extractor, limit=2, chunk_size=2, verbose=False)
    assert stats1.total_matched == 3
    assert stats1.already_extracted == 0
    assert stats1.newly_extracted == 2

    assert p_stapler.count_extracted() == 2

    # Step 2: Verify p_balloon is completely independent (0 extracted)
    assert p_balloon.count_extracted() == 0

    # Step 3: Run p_stapler again; it should extract ONLY the delta record R3
    stats2 = p_stapler.orchestrate_pipeline(extractor=extractor, verbose=False)
    assert stats2.already_extracted == 2
    assert stats2.newly_extracted == 1
    assert p_stapler.count_extracted() == 3

    # Step 4: Run p_balloon; all 3 records are unextracted for p_balloon
    stats_b = p_balloon.orchestrate_pipeline(extractor=extractor, limit=1, verbose=False)
    assert stats_b.already_extracted == 0
    assert stats_b.newly_extracted == 1
    assert p_balloon.count_extracted() == 1

    # Step 5: Raw metadata lookup from project
    meta = p_stapler.get_raw_metadata("R1")
    assert meta is not None
    assert meta["brand_name"] == "DevA"
    assert meta["event_type"] == "Malfunction"

    # Step 6: Project Vigipy export
    vdf = p_stapler.to_vigipy_df()
    assert len(vdf) > 0
    assert "product" in vdf.columns

    export_path = p_stapler.export_vigipy_dataset("staplers.parquet")
    assert os.path.exists(export_path)

    # Check project stats
    p_stats = p_stapler.stats()
    assert p_stats["name"] == "staplers"
    assert p_stats["total_extracted_reports"] == 3
    assert p_stats["total_exports"] == 1

    p_stapler.close()
    p_balloon.close()
