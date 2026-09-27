"""
Unit tests for Gemma 4 fallback engine and Gemini 3.8 Flash distillation interface.
"""

from spotlight.llm_fallback import Gemma4FallbackEngine, Gemini38FlashDistillationClient
from spotlight.schemas import CategoryType, TemporalTiming


def test_parse_llm_json_markdown():
    engine = Gemma4FallbackEngine()
    raw_markdown = """
    Here is the extraction result:
    ```json
    {
      "findings": [
        {
          "category": "OPERATIONAL_PROBLEM",
          "verbatim_span": "balloon burst during inflation",
          "temporal_timing": "INTRA_USE",
          "confidence": 0.95
        }
      ]
    }
    ```
    """
    items = engine._parse_llm_json(raw_markdown)
    assert len(items) == 1
    assert items[0]["category"] == "OPERATIONAL_PROBLEM"
    assert items[0]["verbatim_span"] == "balloon burst during inflation"


def test_gemma4_extraction_and_grounding():
    engine = Gemma4FallbackEngine()
    narrative = "Prior to surgery, a packaging defect identified prior to surgery was noted. Device locked up during procedure."
    findings = engine.extract_with_gemma4(narrative, use_mock=True)

    categories = [f.category for f in findings]
    assert CategoryType.MANUFACTURING_ISSUE in categories
    assert CategoryType.OPERATIONAL_PROBLEM in categories

    mfg = [f for f in findings if f.category == CategoryType.MANUFACTURING_ISSUE][0]
    assert mfg.temporal_timing == TemporalTiming.PRE_USE
    # Check ontology grounding
    assert len(mfg.normalized_terms) > 0
    assert mfg.normalized_terms[0].ontology == "IMDRF_Annex_C"


def test_gemini_38_flash_distillation_prompt():
    client = Gemini38FlashDistillationClient()
    narrative = "Patient underwent angioplasty. Catheter tip broke."
    prompt = client.format_distillation_prompt(narrative)

    assert "gemini-3.8-flash" == client.model_name
    assert "OPERATIONAL_PROBLEM" in prompt
    assert "MANUFACTURING_ISSUE" in prompt
    assert "chain-of-thought" in prompt
    assert narrative in prompt
