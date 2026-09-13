"""Runtime configuration.

Everything the harness needs to be re-tuned without a code change lives here:
model IDs, retry counts, backoff schedule, timeouts, and the dataset location.
Values come from environment variables or a local `.env` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- Dataset (phase 2) -------------------------------------------------
    # Points at the Jeopardy clue TSV. Not used by the general-question
    # endpoint; wired now so the dataset work has a single seam to plug into.
    dataset_path: Path = Path("data/jeopardy_sample.tsv")

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
