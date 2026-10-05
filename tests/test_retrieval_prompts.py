import json

from doc_retrieval import load_glossary, rrf_fuse
from rag_sql import EXAMPLE_BANK_PATH, _with_own_table, build_messages


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
