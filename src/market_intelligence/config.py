"""Application settings, loaded from environment variables and a local .env file."""
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
    news_request_timeout: float = 15.0
    competitor_request_timeout: float = 15.0
    num_articles: int = 2


settings = Settings()
