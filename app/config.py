"""Runtime configuration.

Everything the harness needs to be re-tuned without a code change lives here:
model IDs, retry counts, backoff schedule, timeouts, and the dataset location.
Values come from environment variables or a local `.env` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.prompts import DEFAULT_PROMPT_NAME, prompt_names


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials -------------------------------------------------------
    # Optional so the app still boots (and reports clearly) with one or zero
    # keys configured. A provider without a key is skipped, not crashed on.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Models ------------------------------------------------------------
    # Model IDs are config, never literals in the call sites: providers rename
    # and retire models faster than we want to cut releases.
    openai_model: str = "gpt-5.5"
    anthropic_model: str = "claude-opus-5"

    # --- Harness: retry and fallback --------------------------------------
    # `max_attempts` is total tries per provider, not retries-after-the-first.
    # `backoff_seconds` is the gap *between* attempts, so it needs at least
    # max_attempts-1 entries; the last value repeats if it is short.
    max_attempts: int = 3
    backoff_seconds: list[float] = Field(default_factory=lambda: [10.0, 20.0])

    # Per-request ceiling for a single provider call.
    request_timeout_seconds: float = 90.0

    # Provider order. First entry is primary; the rest are fallbacks in order.
    provider_order: list[str] = Field(default_factory=lambda: ["openai", "anthropic"])

    # --- Harness: behaviour toggles ---------------------------------------
    # When true, a non-retryable error (bad key, unknown model, malformed
    # request) fails over immediately instead of burning the backoff schedule.
    classify_errors: bool = True

    # Enables the `force_fail` request field used to demo the fallback chain.
    # Leave on for local dev and demos; turn off for anything public.
    fault_injection_enabled: bool = True

    # --- Prompt ------------------------------------------------------------
    # Which named variant from app/prompts.py the engines use. Config rather
    # than a literal so an eval run can swap the prompt without a code change
    # -- the same reason model IDs are config.
    prompt_variant: str = DEFAULT_PROMPT_NAME

    # --- Observability -----------------------------------------------------
    # Tracing is off unless explicitly enabled *and* credentialed. A missing
    # or broken tracer must never affect a response: see app/obs/base.py.
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"

    # --- Dataset (phase 2) -------------------------------------------------
    # Points at the Jeopardy clue TSV. No clue data ships with this repo, so
    # this path is routinely absent -- that is a supported state, not an error.
    dataset_path: Path = Path("data/jeopardy_sample.tsv")

    # --- Retrieval ---------------------------------------------------------
    # Off by default: the index is not wired into the prompt yet, and a fresh
    # clone has no clue data to build one from.
    retrieval_enabled: bool = False

    # Where the Chroma store lives. MUST stay gitignored: Chroma keeps the
    # document text next to each vector, so committing it would commit the
    # clue data this repo deliberately does not redistribute.
    index_path: Path = Path("data/chroma")
    collection_name: str = "jeopardy_clues"

    # How a row becomes embedded text.
    #   qa-glued   clue + response in one chunk (the assignment's brief)
    #   clue-only  clue text only; honest control for retrieval evaluation,
    #              since qa-glued puts the answer inside the indexed text
    chunk_scheme: str = "qa-glued"

    # `hash` is offline, free, deterministic and meaningless -- it exists so
    # the pipeline can be tested without a key. `openai` is the real one.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Multi-agent system (ADK, session 3) -------------------------------
    # Entirely separate from the FastAPI harness above. ADK's Runner owns its
    # own orchestration, so it deliberately does not share the retry/backoff
    # machinery -- see README "Multi-agent system".
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"

    # SQLite export of the clue TSV, queried by the MCP server. Gitignored for
    # the same reason as the vector index: it is the clue data.
    clues_db_path: Path = Path("data/clues.sqlite3")

    # Where the A2A judge agent listens, and where the router looks for it.
    judge_agent_url: str = "http://localhost:8001"

    # --- Server ------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("backoff_seconds")
    @classmethod
    def _non_empty_backoff(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("backoff_seconds must contain at least one value")
        if any(s < 0 for s in v):
            raise ValueError("backoff_seconds values must be non-negative")
        return v

    @field_validator("max_attempts")
    @classmethod
    def _at_least_one_attempt(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v

    @field_validator("prompt_variant")
    @classmethod
    def _known_prompt(cls, v: str) -> str:
        # Fail at startup, not at request time: a typo that silently fell back
        # to the default would make an eval run report the production rate
        # under the control's name.
        if v not in prompt_names():
            raise ValueError(f"unknown prompt_variant {v!r}; known: {prompt_names()}")
        return v

    @field_validator("chunk_scheme")
    @classmethod
    def _known_chunk_scheme(cls, v: str) -> str:
        from app.retrieval.chunks import CHUNK_SCHEMES

        if v not in CHUNK_SCHEMES:
            raise ValueError(f"unknown chunk_scheme {v!r}; known: {list(CHUNK_SCHEMES)}")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _known_embedding_provider(cls, v: str) -> str:
        known = {"openai", "hash"}
        if v not in known:
            raise ValueError(f"unknown embedding_provider {v!r}; known: {sorted(known)}")
        return v

    @field_validator("provider_order")
    @classmethod
    def _known_providers(cls, v: list[str]) -> list[str]:
        known = {"openai", "anthropic"}
        unknown = [p for p in v if p not in known]
        if unknown:
            raise ValueError(f"unknown provider(s) {unknown}; known: {sorted(known)}")
        if not v:
            raise ValueError("provider_order must not be empty")
        return v

    def backoff_for(self, attempt_index: int) -> float:
        """Seconds to sleep *before* attempt `attempt_index` (1-based).

        Attempt 1 never sleeps. Beyond the configured list the last value
        repeats, so a longer `max_attempts` doesn't need a longer schedule.
        """
        if attempt_index <= 1:
            return 0.0
        gap = attempt_index - 2
        if gap < len(self.backoff_seconds):
            return self.backoff_seconds[gap]
        return self.backoff_seconds[-1]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
