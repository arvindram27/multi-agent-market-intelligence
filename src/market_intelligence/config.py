"""Application settings, loaded from environment variables and a local .env file."""
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the market intelligence ecosystem.

    Values are read from environment variables (e.g. ``OPENAI_MODEL``) and,
    when present, a ``.env`` file in the project root. See ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Network limits. The timeouts are applied by the bounded HTTP client, not
    # just stored as documentation.
    news_request_timeout: float = Field(default=15.0, gt=0)
    competitor_request_timeout: float = Field(default=15.0, gt=0)
    max_feed_bytes: int = Field(default=2_000_000, gt=0)
    max_page_bytes: int = Field(default=1_000_000, gt=0)
    max_redirects: int = Field(default=3, ge=0, le=10)

    # Ingestion and model-output bounds.
    num_articles: int = Field(default=2, ge=1, le=10)
    max_articles: int = Field(default=10, ge=1, le=10)
    min_article_chars: int = Field(default=40, ge=0)
    max_competitor_chars: int = Field(default=8_000, gt=0)

    # Per-call and per-run LLM limits. Costs are not guessed: the run is capped
    # by tokens and calls, and usage is reported separately.
    llm_request_timeout: float = Field(default=30.0, gt=0)
    llm_max_output_tokens: int = Field(default=512, ge=1)
    max_run_tokens: int = Field(default=20_000, ge=1)
    max_run_llm_calls: int = Field(default=24, ge=1)

    @model_validator(mode="after")
    def _default_article_count_is_in_bounds(self):
        if self.num_articles > self.max_articles:
            raise ValueError("NUM_ARTICLES must not exceed MAX_ARTICLES.")
        return self


settings = Settings()
