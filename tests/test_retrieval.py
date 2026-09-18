"""Retrieval plumbing: chunking, the column trap, manifests, and index states.

Offline throughout. The `hash` embedder needs no key and no network, and the
state machine is exercised with real files in tmp_path rather than mocks, so
what is tested is what runs.
"""

from __future__ import annotations

import csv

import pytest

from app.config import Settings
from app.retrieval.base import IndexManifest, UnavailableStore, describe_source
from app.retrieval.chunks import CHUNK_SCHEME_VERSION, read_chunks, render
from app.retrieval.embedders import HashEmbedder, build_embedder
from app.retrieval.index import inspect, open_store

ROWS = [
    {
        "round": "1",
        "clue_value": "100",
        "daily_double_value": "0",
        "category": "LAKES & RIVERS",
        "comments": "",
        "answer": "River mentioned most often in the Bible",
        "question": "the Jordan",
        "air_date": "1984-09-10",
        "notes": "",
    },
    {
        "round": "3",
        "clue_value": "0",
        "daily_double_value": "0",
        "category": "WORLD OF FOOD",
        "comments": "",
        "answer": "Like chop suey, this Chinese sweet was invented in America",
        "question": "the fortune cookie",
        "air_date": "1984-09-12",
        "notes": "",
    },
    {  # missing a response: unretrievable, must be skipped
        "round": "1",
        "clue_value": "200",
        "daily_double_value": "0",
        "category": "BROKEN",
        "comments": "",
        "answer": "a clue with no response",
        "question": "",
        "air_date": "1984-09-13",
        "notes": "",
    },
]


@pytest.fixture
def tsv(tmp_path):
    path = tmp_path / "clues.tsv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROWS[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(ROWS)
    return path


@pytest.fixture
def rag_settings(tmp_path, tsv):
    return Settings(
        openai_api_key="test-key",
        retrieval_enabled=True,
        dataset_path=tsv,
        index_path=tmp_path / "chroma",
        embedding_provider="hash",
        embedding_dimensions=64,
    )


# --------------------------------------------------------------------------
# The column trap
# --------------------------------------------------------------------------
def test_inverted_columns_are_renamed_at_the_boundary(tsv):
    """`answer` is the clue and `question` is the response. Fix it once, here."""
    chunks = list(read_chunks(tsv))
    first = chunks[0]
    assert first.clue_text == "River mentioned most often in the Bible"
    assert first.correct_response == "the Jordan"


def test_rows_missing_either_half_are_skipped(tsv):
    chunks = list(read_chunks(tsv))
    assert len(chunks) == 2
    assert all(c.clue_text and c.correct_response for c in chunks)
    assert "BROKEN" not in {c.category for c in chunks}


def test_round_numbers_become_names(tsv):
    chunks = list(read_chunks(tsv))
    assert chunks[0].round == "Jeopardy"
    assert chunks[1].round == "Final Jeopardy"


def test_wrong_file_fails_with_a_useful_message(tmp_path):
    path = tmp_path / "wrong.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        list(read_chunks(path))


# --------------------------------------------------------------------------
# Chunk schemes
# --------------------------------------------------------------------------
def test_qa_glued_includes_the_response_clue_only_does_not():
    """The distinction that keeps a retrieval evaluation honest.

    qa-glued puts the correct response inside the embedded text, so querying
    with clue text scores well partly because the answer was indexed.
    """
    kwargs = {"clue_text": "a clue", "correct_response": "a response", "category": "CAT"}
    glued = render(**kwargs, scheme="qa-glued")
    clue_only = render(**kwargs, scheme="clue-only")
    assert "a response" in glued
    assert "a response" not in clue_only
    assert "a clue" in clue_only


def test_response_is_always_on_the_chunk_as_metadata(tsv):
    """Even under clue-only, so evaluation can check correctness."""
    chunk = next(read_chunks(tsv, scheme="clue-only"))
    assert chunk.metadata["correct_response"] == "the Jordan"
    assert "the Jordan" not in chunk.text


def test_unknown_scheme_raises(tsv):
    with pytest.raises(ValueError, match="unknown chunk scheme"):
        list(read_chunks(tsv, scheme="nope"))


def test_settings_rejects_unknown_scheme_and_embedder():
    with pytest.raises(ValueError, match="unknown chunk_scheme"):
        Settings(chunk_scheme="nope")
    with pytest.raises(ValueError, match="unknown embedding_provider"):
        Settings(embedding_provider="nope")


def test_chunk_ids_are_content_derived_and_stable(tsv):
    """IDs must survive regenerating a sample with a different stride."""
    first = [c.id for c in read_chunks(tsv)]
    second = [c.id for c in read_chunks(tsv)]
    assert first == second
    assert len(set(first)) == len(first)


def test_limit_stops_early(tsv):
    assert len(list(read_chunks(tsv, limit=1))) == 1


# --------------------------------------------------------------------------
# Manifest / staleness
# --------------------------------------------------------------------------
def _manifest(**overrides) -> IndexManifest:
    base = {
        "source_path": "data/x.tsv",
        "source_size": 100,
        "source_mtime_ns": 1,
        "row_count": 10,
        "embedder_id": "hash:64",
        "dimensions": 64,
        "chunk_scheme": "qa-glued",
        "chunk_scheme_version": CHUNK_SCHEME_VERSION,
    }
    return IndexManifest(**{**base, **overrides})


def test_manifest_roundtrips(tmp_path):
    m = _manifest()
    m.write(tmp_path)
    assert IndexManifest.read(tmp_path) == m


def test_missing_manifest_reads_as_none(tmp_path):
    assert IndexManifest.read(tmp_path) is None


def test_corrupt_manifest_reads_as_none_rather_than_raising(tmp_path):
    (tmp_path / "index_manifest.json").write_text("{not json", encoding="utf-8")
    assert IndexManifest.read(tmp_path) is None


def test_row_count_alone_is_not_staleness():
    """row_count is an output of building, not an input to it."""
    assert _manifest(row_count=10).differences(_manifest(row_count=999)) == []


def test_same_dimensions_different_model_is_detected():
    """The silent killer: swapping models at equal width returns garbage, not an error."""
    changes = _manifest(embedder_id="openai:a:64").differences(_manifest(embedder_id="openai:b:64"))
    assert len(changes) == 1
    assert "embedder_id" in changes[0]


def test_scheme_change_is_detected():
    assert _manifest(chunk_scheme="clue-only").differences(_manifest(chunk_scheme="qa-glued"))


def test_describe_source_changes_when_the_file_does(tmp_path):
    path = tmp_path / "f.tsv"
    path.write_text("a", encoding="utf-8")
    before = describe_source(path)
    path.write_text("aa", encoding="utf-8")
    assert describe_source(path) != before


# --------------------------------------------------------------------------
# Index states -- the three from the design, plus stale
# --------------------------------------------------------------------------
def test_no_dataset_is_reported_not_raised(tmp_path):
    """A fresh clone has no clue data. The app must stay up and say so."""
    settings = Settings(
        retrieval_enabled=True,
        dataset_path=tmp_path / "absent.tsv",
        index_path=tmp_path / "chroma",
        embedding_provider="hash",
    )
    state = inspect(settings)
    assert state.status == "no_dataset"
    assert "make sample" in state.reason


def test_dataset_present_but_no_index_needs_build(rag_settings):
    assert inspect(rag_settings).status == "needs_build"


def test_missing_key_for_openai_embedder_is_reported(tmp_path, tsv):
    settings = Settings(
        retrieval_enabled=True,
        dataset_path=tsv,
        index_path=tmp_path / "chroma",
        embedding_provider="openai",
    )
    state = inspect(settings)
    assert state.status == "unavailable"
    assert "OPENAI_API_KEY" in state.reason


def test_stale_index_is_detected(rag_settings):
    stale = IndexManifest(
        source_path=str(rag_settings.dataset_path),
        source_size=1,
        source_mtime_ns=1,
        row_count=2,
        embedder_id="hash:64",
        dimensions=64,
        chunk_scheme="qa-glued",
        chunk_scheme_version=CHUNK_SCHEME_VERSION,
    )
    stale.write(rag_settings.index_path)
    state = inspect(rag_settings)
    assert state.status == "stale"
    assert any("source_size" in c for c in state.changes or [])


def test_disabled_retrieval_yields_an_explaining_store(rag_settings):
    settings = rag_settings.model_copy(update={"retrieval_enabled": False})
    store = open_store(settings)
    assert isinstance(store, UnavailableStore)
    assert store.is_available() == "RETRIEVAL_ENABLED is false"


def test_unavailable_store_returns_empty_rather_than_raising():
    """A missing index must degrade the answer, not fail the request."""
    store = UnavailableStore("nope")
    assert store.search("anything") == []
    assert store.count() == 0


def test_unbuilt_index_tells_you_to_build_it(rag_settings):
    store = open_store(rag_settings)
    assert "make index" in (store.is_available() or "")


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(64)
    first = e.embed(["a clue"])[0]
    assert first == e.embed(["a clue"])[0]
    assert len(first) == 64
    assert abs(sum(v * v for v in first) ** 0.5 - 1.0) < 1e-9


def test_hash_embedder_separates_different_texts():
    e = HashEmbedder(64)
    a, b = e.embed(["one", "two"])
    assert a != b


def test_build_embedder_honours_config(rag_settings):
    assert isinstance(build_embedder(rag_settings), HashEmbedder)
    assert build_embedder(rag_settings).id == "hash:64"
