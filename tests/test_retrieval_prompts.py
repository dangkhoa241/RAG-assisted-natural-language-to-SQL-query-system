import json

from doc_retrieval import load_glossary, rrf_fuse
from rag_sql import EXAMPLE_BANK_PATH, _with_own_table, build_messages, llm_params


def bank():
    return [_with_own_table(json.loads(line)) for line in EXAMPLE_BANK_PATH.read_text(encoding="utf-8").splitlines()]


def test_every_bank_example_uses_its_own_table():
    for e in bank():
        assert "FROM data" not in e["table_sql"]
        assert f"FROM {e['table']}" in e["table_sql"]
        assert e["table_schema"].startswith(f"Table `{e['table']}` columns: `")


def test_fixed_prompt_shows_example_tables_and_definitions():
    examples = bank()[:2]
    docs = [{"term": "Premium insurer", "definition": "Aetna or Cigna."}]
    prompt = build_messages("q", "SCHEMA", examples, docs)[1]["content"]
    assert "Business definitions" in prompt and "- Premium insurer: Aetna or Cigna." in prompt
    assert f"FROM {examples[0]['table']}" in prompt and "FROM data" not in prompt


def test_prompt_without_context_matches_zero_shot_format():
    assert build_messages("q", "SCHEMA", [], [])[1]["content"] == "Schema:\nSCHEMA\n\nQuestion: q\nSQL:"


def test_glossaries_chunk_by_definition_with_unique_ids():
    for name, prefix in [("healthcare", "hc_"), ("retail", "rt_")]:
        chunks = load_glossary(name)
        assert 25 <= len(chunks) <= 30
        assert len({c["id"] for c in chunks}) == len(chunks)
        assert all(c["id"].startswith(prefix) and c["definition"] for c in chunks)


def test_rrf_prefers_items_ranked_well_by_both():
    assert rrf_fuse([[0, 1, 2], [1, 2, 0]])[0] == 1


def test_cache_key_separates_providers_and_keeps_groq_keys():
    import hashlib

    from rag_sql import LLMCache, cache_key

    messages = build_messages("q", "SCHEMA", [], [])
    params = llm_params("openai/gpt-oss-120b")
    old_formula = json.dumps({"model": "openai/gpt-oss-120b", "messages": messages, "params": params}, sort_keys=True)
    assert cache_key(messages, "openai/gpt-oss-120b") == hashlib.sha256(old_formula.encode("utf-8")).hexdigest()
    assert (LLMCache.make_key("gpt-oss-120b", messages, params, "groq")
            != LLMCache.make_key("gpt-oss-120b", messages, params, "cerebras"))


def test_gpt_oss_params_are_the_same_on_both_providers():
    assert llm_params("gpt-oss-120b") == llm_params("openai/gpt-oss-120b") == {
        "temperature": 0, "max_completion_tokens": 1024, "reasoning_effort": "low"}


def test_aliases_stay_out_of_the_definition_text():
    for name in ("healthcare", "retail", "saas"):
        for c in load_glossary(name):
            assert "Aliases:" not in c["definition"] and "Aliases:" not in c["text"]
            assert c["aliases"][0] == c["aliases"][0].strip() and len(c["aliases"]) >= 2
    aov = next(c for c in load_glossary("retail") if c["id"] == "rt_average_order_value")
    assert {"Average order value", "AOV"} <= set(aov["aliases"])


def test_term_gating_matches_whole_phrases_and_prefers_the_longer_match():
    from doc_retrieval import match_terms

    hc = load_glossary("healthcare")
    ids = lambda question: [c["id"] for c in match_terms(question, hc)]
    assert ids("how many flagged claims are there?") == ["hc_flagged_claim"]
    assert ids("admissions with flagged test results") == ["hc_flagged_result"]
    assert ids("average billing of senior care tier patients") == ["hc_senior_care_tier"]
    assert ids("admissions by fiscal quarter in FY2022") == ["hc_fiscal_year", "hc_fiscal_quarter"]
    assert ids("average length of stay by admission type") == []
    assert ids("seniority of the ICU staff") == ["hc_icu_room"]  # "senior" must be a whole word


def test_gated_prompt_without_matches_is_the_zero_shot_prompt():
    from doc_retrieval import match_terms

    docs = match_terms("average billing amount by gender", load_glossary("healthcare"))
    assert docs == [] and build_messages("q", "S", [], docs) == build_messages("q", "S", [], [])
